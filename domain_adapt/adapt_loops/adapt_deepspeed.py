import json
import os.path as osp
import time
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import copy
from mmengine.logging import print_log

try:
    import deepspeed
except ImportError:
    deepspeed = None

import torch.nn as nn
from mmengine.registry import (MODEL_WRAPPERS, OPTIM_WRAPPERS, OPTIMIZERS,
                               STRATEGIES)
from mmengine.optim import BaseOptimWrapper, _ParamScheduler
from mmengine.runner.checkpoint import save_checkpoint, weights_to_cpu
from mmengine.utils import apply_to, digit_version, get_git_hash
from mmengine._strategy.deepspeed import DeepSpeedStrategy, DeepSpeedOptimWrapper
from collections import OrderedDict



@STRATEGIES.register_module()
class DeepSpeedStrategy_adapt(DeepSpeedStrategy):
    def __init__(
        self,
        *,
        # the following args are for deepspeed
        config: Union[str, dict, None] = None,
        zero_optimization: Optional[dict] = None,
        gradient_clipping: Optional[float] = None,
        fp16: Optional[dict] = None,
        inputs_to_half: Optional[List[Union[int, str]]] = None,
        bf16: Optional[dict] = None,
        amp: Optional[dict] = None,
        activation_checkpointing: Optional[dict] = None,
        aio: Optional[dict] = None,
        train_micro_batch_size_per_gpu: Optional[int] = None,
        gradient_accumulation_steps: Optional[int] = None,
        # disable the log printed by deepseed
        steps_per_print: int = 10000000000000,
        # the following args are for BaseStrategy
        exclude_frozen_parameters: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(config=config, zero_optimization=zero_optimization, 
                         gradient_clipping=gradient_clipping, fp16=fp16, 
                         inputs_to_half=inputs_to_half, bf16=bf16, 
                         amp=amp, activation_checkpointing=activation_checkpointing, 
                         aio=aio, train_micro_batch_size_per_gpu=train_micro_batch_size_per_gpu,
                         gradient_accumulation_steps=gradient_accumulation_steps, 
                         steps_per_print=steps_per_print, exclude_frozen_parameters=exclude_frozen_parameters, **kwargs)

    def prepare(
        self,
        model: Union[nn.Module, dict],
        *,
        optim_wrapper: Union[BaseOptimWrapper, dict, None] = None,
        param_scheduler: Union[_ParamScheduler, Dict, List, None] = None,
        compile: Union[dict, bool] = False,
        dispatch_kwargs: Optional[dict] = None,
    ):
        if self._prepared:
            return self._prepared_components()
        assert dispatch_kwargs is not None
        self.dispatch_kwargs.update(dispatch_kwargs)

        model = self.build_model(model)
        model = self._init_model_weights(model)


        # teacher model
        import copy
        with torch.no_grad():
            teacher_model = copy.deepcopy(model) 
        teacher_model = teacher_model.half().to('cuda')  
        teacher_model.requires_grad_(False)

        if optim_wrapper is not None:
            self.optim_wrapper = self.build_optim_wrapper(optim_wrapper, model)
            self.model = self._wrap_model(model, teacher_model)

            self.optim_wrapper.model = self.model  # type: ignore

        else:
            self.model = self._wrap_model(model)

        if param_scheduler is not None:
            self.param_schedulers = self.build_param_scheduler(
                param_scheduler, self.optim_wrapper)
        self._prepared = True
        return self._prepared_components()


    def _wrap_model(self, model: nn.Module, teacher_model: nn.Module,) -> nn.Module:
        if hasattr(self, 'optim_wrapper'):
            engine, self.optim_wrapper.optimizer, *_ = deepspeed.initialize(
                model=model,
                optimizer=self.optim_wrapper.optimizer,
                config=self.config)
        else:
            engine, *_ = deepspeed.initialize(model=model, config=self.config)

        wrapper = MMDeepSpeedEngineWrapper_adapt(
            model=engine, inputs_to_half=self._inputs_to_half, teacher_model=teacher_model)
        return wrapper


@MODEL_WRAPPERS.register_module()
class MMDeepSpeedEngineWrapper_adapt:

    def __init__(
        self,
        *,
        model: 'deepspeed.DeepSpeedEngine',
        inputs_to_half: Optional[List[Union[int, str]]] = None,
        teacher_model: Optional['nn.Module'] = None,
    ):
        self.model = model
        self.teacher_model = teacher_model
        self._inputs_to_half = inputs_to_half

    def __getattr__(self, name):
        return getattr(self.model, name)

    def train_step(
        self,
        data: Union[dict, tuple, list],
        optim_wrapper: DeepSpeedOptimWrapper,
    ) -> Dict[str, torch.Tensor]:
        

        # Source Domain
        self.model.module.state = 'source'
        # data_processing
        data_source = self.process_img_resize(data['source'], mode='source')
        data_source = self.model.module.data_preprocessor(data_source, training=True)
        data_source = self._cast_inputs_half(data_source)
        # model forward
        losses = self._run_forward(data_source, mode='loss')         
        source_parsed_loss, source_log_vars = self.model.module.parse_losses(losses) 
        # model backward
        optim_wrapper.backward(source_parsed_loss, retain_graph=True)



        # # Target Teacher Domain
        self.teacher_model.state = 'teacher'
        # data_processing
        data_teacher = copy.deepcopy(data['target'])
        data_teacher = self.process_img_resize(data_teacher, mode='teacher')   
        data_teacher = self.model.module.data_preprocessor(data_teacher, training=True) 
        data_teacher = self._cast_inputs_half(data_teacher)
        # teacher_model forward
        with torch.no_grad():  
            prototype, teacher_feat = self.teacher_model(data_teacher['inputs'], data_teacher['data_samples'], mode='loss') 



        # # Target Domain
        self.model.module.state = 'targetS'
        # data_processing
        data_target = self.process_img_resize(data['target'], mode='target')
        data_target = self.model.module.data_preprocessor(data_target, training=True)
        data_target = self._cast_inputs_half(data_target)
        # data_processing S
        data_targetS = copy.deepcopy(data['target'])
        data_targetS = self.process_img_aug(data_targetS, mode='targetS')
        data_targetS = self.model.module.data_preprocessor(data_targetS, training=True)
        data_targetS = self._cast_inputs_half(data_targetS)
        
        # input target Strong data
        for i in range(len(data_targetS['inputs'])):
            data_target['data_samples'][i].img_aug = data_targetS['inputs'][i]
        del data_targetS
        # torch.cuda.empty_cache() 
        # input teacher_feat and prototype to the model
        data_target['data_samples'][0].teacher_feat = teacher_feat     
        data_target['data_samples'][0].objective_vectors = prototype
        # model forward
        target_losses = self._run_forward(data_target, mode='loss')
        target_parsed_loss, target_log_vars = self.model.module.parse_losses(target_losses)
        # model backward
        optim_wrapper.backward(target_parsed_loss)




        # update teacher model using EMA           
        model_params = dict(self.model.module.named_parameters())  
        teacher_params = dict(self.teacher_model.named_parameters()) 
        for name, param_k in teacher_params.items():  
            if name in model_params:  
                param_q = model_params[name]
                param_k.data.mul_(0.999).add_(param_q.data, alpha=(1 - 0.999))  # EMA update
        
        # synchronize buffers like BatchNorm between student and teacher
        model_buffers = dict(self.model.module.named_buffers())
        teacher_buffers = dict(self.teacher_model.named_buffers())
        for name, buffer_k in teacher_buffers.items():
            if name in model_buffers:
                buffer_k.data.copy_(model_buffers[name].data)

        # update parameters
        optim_wrapper.step()

        # loss recording
        combined_logs = OrderedDict()
        for key, value in source_log_vars.items():
            combined_logs[f"source/{key}"] = value

        for key, value in target_log_vars.items():
            combined_logs[f"target/{key}"] = value

        return combined_logs

    def val_step(self, data: Union[dict, tuple, list]) -> list:
        """Gets the prediction of module during validation process.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        self.model.module.stage = 'stage_predict'
        data = self.process_img_resize(data, mode='valid')
        data = self.model.module.data_preprocessor(data, False)
        data = self._cast_inputs_half(data)
        return self._run_forward(data, mode='predict')

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        """Gets the predictions of module during testing process.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.model.module.data_preprocessor(data, False)
        data = self._cast_inputs_half(data)
        return self._run_forward(data, mode='predict')

    def _run_forward(self, data: Union[dict, tuple, list], mode: str) -> Any:
        """Unpacks data for :meth:`forward`

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            mode (str): Mode of forward.

        Returns:
            dict or list: Results of training or testing mode.
        """
        if isinstance(data, dict):
            results = self.model(**data, mode=mode)
        elif isinstance(data, (list, tuple)):
            results = self.model(*data, mode=mode)
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            f'list, tuple or dict, but got {type(data)}')
        return results

    def _cast_inputs_half(self, inputs: Union[list, tuple, dict, None]):
        """Cast inputs to half precision if needed.

        Args:
            inputs (list or tuple or dict or None): Inputs to be casted.

        Returns:
            list or tuple or dict or None: Casted inputs.
        """
        if self._inputs_to_half is None:
            return inputs

        dtype = next(self.model.parameters()).dtype
        if isinstance(inputs, (list, tuple)):
            new_inputs = []
            for i, v in enumerate(inputs):
                if i in self._inputs_to_half:
                    new_inputs.append(
                        apply_to(v, lambda x: hasattr(x, 'to'),
                                 lambda x: x.to(dtype)))
                else:
                    new_inputs.append(v)
            return inputs.__class__(new_inputs)
        elif isinstance(inputs, dict):
            for k, v in inputs.items():
                if k in self._inputs_to_half:
                    inputs[k] = apply_to(v, lambda x: hasattr(x, 'to'),
                                         lambda x: x.to(dtype))
            return inputs
        else:
            raise TypeError('inputs should be list, tuple or dict, '
                            f'but got {type(inputs)}')
        

    def process_img_resize(self, dataset, mode=None): 
        missing_resize = False

        for i in range(len(dataset['data_samples'])):
            if hasattr(dataset['data_samples'][i], 'img_resize'):
                if mode=='teacher':
                    dataset['inputs'][i] = dataset['data_samples'][i].img_resize
                del dataset['data_samples'][i].img_resize   
                torch.cuda.empty_cache()
            else:
                missing_resize = True
        if missing_resize:
            print(f"Warning: Some data_samples are missing 'img_resize'.")
        return dataset
    

    def process_img_aug(self, dataset, mode=None): 

        # not every dataset needs to have img_aug
        for i in range(len(dataset['data_samples'])):
            if hasattr(dataset['data_samples'][i], 'img_aug'):
                if mode=='targetS':
                    dataset['inputs'][i] = dataset['data_samples'][i].img_aug
                del dataset['data_samples'][i].img_aug
        return dataset
