You do not need root or administrative privileges to set up your deep learning and voice science pipeline. In fact, creating an isolated virtual environment is the industry standard and best practice for this exact scenario.When you don't have root access, you cannot modify the server's global system files (/usr/bin or /usr/lib). A virtual environment solves this completely by building a self-contained, isolated directory right inside your home folder (/home/username/), where you have absolute read and write permissions.🛠️ Step-by-Step Implementation GuideSince you verified earlier that python3 is available on your server, you can use its built-in venv module to generate your sandbox workspace.Step 1: Navigate and Create the Environment WorkspaceConnect to your server via the terminal, navigate to your project's root folder, and generate a localized environment directory (traditionally named .venv):Bash# Navigate to your speech repository folder
cd ~/vocal-resonance-analysis

# Generate the isolated environment folder structure
python3 -m venv .venv
This command copies the python3 binary and the pip package manager into a new hidden directory called .venv inside your project folder.Step 2: Activate the Local WorkspaceTo tell your current terminal session to bypass the server's global system and use your local folder instead, you must activate it:Bashsource .venv/bin/activate
Once executed, your terminal prompt will change to look something like this:Bash(.venv) user@server:~/vocal-resonance-analysis$
⚠️ Important Reminder: Every single time you log into the server or open a new terminal tab in VS Code, you must rerun source .venv/bin/activate to enter your workspace.Step 3: Install Your Dependencies LocallyNow that the environment is active, running pip3 or python3 -m pip will automatically download and unpack libraries directly into your local .venv folder, without ever prompting you for a root password:Bash# Upgrade your local pip installer tool first
python3 -m pip install --upgrade pip

# Install the exact dependencies for your 300ms vowel feature extraction pipeline
python3 -m pip install torch transformers librosa pandas numpy tqdm scikit-learn
📓 Linking Your New Environment to VS Code Jupyter NotebooksOnce your package installation finishes in the terminal, you need to make sure your .ipynb notebook canvas knows where to look:Open your .ipynb file in VS Code.Click Select Kernel in the top-right corner of the editor canvas screen.Choose Python Environments... $\rightarrow$ select the path that points explicitly to the environment inside your project folder (it will show up with a relative path like ./venv/bin/python or matching your project directory name).Run your first cell block containing import torch and import librosa. It will now load successfully with zero permission errors.

(.venv) [yuanyuan@puhti-login11 resonant-singing-voice-analysis]$ python3 -m pip install ipykernel

(.venv) [yuanyuan@puhti-login11 resonant-singing-voice-analysis]$ python3 -m ipykernel install --user --name=singing-resonance --display-name "Singing Voice Resonance Lab"
Installed kernelspec singing-resonance in /users/yuanyuan/.local/share/jupyter/kernels/singing-resonance

