import utils.virt_env as venv

pip_path = venv.activate_env()
output = venv.env_kernel(pip_path, kernel_name='halal_food_classifier')