

python deploy.py --config-name=idp3.yaml \
                            task=gr1_dex-3d \
                            hydra.run.dir=data/outputs/gr1_dex-3d-idp3-0913_example_seed0 \
                            training.debug=False \
                            training.seed=0 \
                            training.device="cuda:0" \
                            exp_name=gr1_dex-3d-idp3-0913_example \
                            logging.mode= \
                            checkpoint.save_ckpt=True \
                            task.dataset.zarr_path=/home/ze/projects/Improved-3D-Diffusion-Policy/training_data_example




