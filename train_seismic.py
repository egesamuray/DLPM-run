#!/usr/bin/env python
import os
import sys
import yaml

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
os.environ.setdefault("NEPTUNE_MODE", "offline")
os.environ.setdefault("NEPTUNE_DISABLE_TELEMETRY", "1")

import bem.Experiments as Exp
from bem.utils_exp import FileHandler
import dlpm.dlpm_experiment as dlpm_exp


def main():
    config_file = os.path.join('dlpm', 'configs', 'seismic_rect.yml')
    p = FileHandler.get_param_from_config(os.path.dirname(config_file), os.path.basename(config_file))

    checkpoint_dir = os.path.join('models', 'seismic_exp')
    exp = Exp.Experiment(
        checkpoint_dir=checkpoint_dir,
        p=p,
        logger=None,
        exp_hash=dlpm_exp.exp_hash,
        eval_hash=None,
        init_method_by_parameter=dlpm_exp.init_method_by_parameter,
        init_models_by_parameter=dlpm_exp.init_models_by_parameter,
        reset_models=dlpm_exp.reset_models,
    )
    exp.prepare()
    exp.run(progress=p['run']['progress'])
    print(exp.save(curr_epoch=p['run']['epochs']))
    exp.terminate()


if __name__ == "__main__":
    main()
