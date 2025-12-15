# import os, pathlib
# import sys, subprocess

# root_path = os.path.join(pathlib.Path(__file__).parent.parent.parent.absolute(), '.env') # path to virtual environment
# # print(root_path) // /home/roxel/snow/02-code/uni/cv-ccp/halal_food_classifier/.env/

# # check if virt env exists
# def env_exist():
#     return os.path.exists(os.path.join(root_path, 'bin', 'activate')) or \
#         os.path.exists(os.path.join(root_path, 'Scripts', 'activate'))

# # Get path to pip in virt env, create virt env if it does not exist
# def activate_env() -> str:
#     if env_exist():
#         print('virtual environment exists')
#         try:
#             if os.name == 'nt': # for Windows
#                 pip_path = os.path.join(root_path, 'Scripts', 'pip')
#             else: # for Unix or MacOS
#                 pip_path = os.path.join(root_path, 'bin', 'pip')

#             return pip_path

#         except os.error as e:
#             return f'failed to activate existing virtual environment: {e}'

#     else:
#         print('virtual environment does not exist, activating...')
#         try:
#             subprocess.run([sys.executable, '-m', 'venv', '.env'], check=True)

#             if os.name == 'nt':
#                 pip_path = os.path.join(root_path, 'Scripts', 'pip')
#             else:
#                 pip_path = os.path.join(root_path, 'bin', 'pip')

#             return pip_path

#         except subprocess.CalledProcessError as e:
#             return f'failed to activate created virtual environment: {e}'

# # Get pip version in virt env
# def pip_ver(pip_path: str) -> str:
#     return str(subprocess.run([pip_path, '--version'], capture_output=True))

# # update pip in virt env
# def update_pip(pip_path: str) -> str:
#     return str(subprocess.run([pip_path, 'install', '--upgrade', 'pip'], capture_output=True))

# # install requirements from requirements.txt in virt env
# def install_requirements() -> str:
#     return str(subprocess.run([pip_path, 'install', '-r', requirements_file], capture_output=True))

# def remove_pkg(package_name: str) -> str:
#     return str(subprocess.run([pip_path, 'uninstall', package_name, '-y'], capture_output=True))


# requirements_file = os.path.join(pathlib.Path(__file__).parent.parent.parent.absolute(), 'requirements.txt')

# pip_path = activate_env()

import os, pathlib
import sys, subprocess

root_path = os.path.join(pathlib.Path(__file__).parent.parent.parent.absolute(), '.env') # path to virtual environment
# print(root_path) // /home/roxel/snow/02-code/uni/cv-ccp/halal_food_classifier/.env/

# check if virt env exists
def env_exist():
    return os.path.exists(os.path.join(root_path, 'bin', 'activate')) or \
        os.path.exists(os.path.join(root_path, 'Scripts', 'activate'))

# Get path to pip in virt env, create virt env if it does not exist
def activate_env() -> str:
    if env_exist():
        print('virtual environment exists')
        try:
            if os.name == 'nt': # for Windows
                pip_path = os.path.join(root_path, 'Scripts', 'pip')
            else: # for Unix or MacOS
                pip_path = os.path.join(root_path, 'bin', 'pip')

            return pip_path

        except os.error as e:
            return f'failed to activate existing virtual environment: {e}'

    else:
        print('virtual environment does not exist, activating...')
        try:
            subprocess.run([sys.executable, '-m', 'venv', '.env'], check=True)

            if os.name == 'nt':
                pip_path = os.path.join(root_path, 'Scripts', 'pip')
            else:
                pip_path = os.path.join(root_path, 'bin', 'pip')

            return pip_path

        except subprocess.CalledProcessError as e:
            return f'failed to activate created virtual environment: {e}'

# Get pip version in virt env
def pip_ver(pip_path: str) -> str:
    return str(subprocess.run([pip_path, '--version'], capture_output=True))

# update pip in virt env
def update_pip(pip_path: str) -> str:
    return str(subprocess.run([pip_path, 'install', '--upgrade', 'pip'], capture_output=True))

# install requirements from requirements.txt in virt env
def install_requirements() -> str:
    return str(subprocess.run([pip_path, 'install', '-r', requirements_file], capture_output=True))

def remove_pkg(package_name: str) -> str:
    return str(subprocess.run([pip_path, 'uninstall', package_name, '-y'], capture_output=True))

# Install and configure ipykernel for use in Jupyter notebooks
def env_kernel(pip_path: str, kernel_name: str = 'halal_food_classifier') -> str:
    try:
        # Install ipykernel in the virtual environment
        install_result = subprocess.run([pip_path, 'install', 'ipykernel'], 
                                        capture_output=True, text=True, check=True)
        print(f'ipykernel installed: {install_result.stdout}')
        
        # Get the Python executable path from the virtual environment
        if os.name == 'nt':
            python_path = os.path.join(root_path, 'Scripts', 'python')
        else:
            python_path = os.path.join(root_path, 'bin', 'python')
        
        # Register the kernel with Jupyter
        kernel_result = subprocess.run([python_path, '-m', 'ipykernel', 'install', 
                                        '--user', '--name', kernel_name, 
                                        '--display-name', f'Python ({kernel_name})'],
                                        capture_output=True, text=True, check=True)
        
        return f'ipykernel configured successfully: {kernel_result.stdout}'
    
    except subprocess.CalledProcessError as e:
        return f'failed to configure ipykernel: {e.stderr}'


requirements_file = os.path.join(pathlib.Path(__file__).parent.parent.parent.absolute(), 'requirements.txt')

pip_path = activate_env()