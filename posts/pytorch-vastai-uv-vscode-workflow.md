I've been playing around more and more with Pytorch and ML workflows recently.
I got to the point where running workloads on my local laptop - M3 Macbook Air - was becoming too slow.

I didn't want to buy a new computer to solve this problem. So I started thinking about an easy and cheap workflow for running ML workloads on cloud instances.

When I’m working on ML projects, I want:

* Clean, reproducible Python environments
* GPUs to speed up model training
* Fast installs
* Local development that feels normal, where I can run Jupyter notebooks and save charts
* Low fees and no ongoing maintenance

After trying a few setups, I landed on a workflow that’s surprisingly simple:

**uv + GitHub + Vast.ai + VS Code Remote SSH**

I feel like I'm working locally, but I can leverage GPUs to speed up training, but only pay when I am training. (And the fees are very small).

Here’s the full setup.

## Stack

### uv

`uv` is a Fast Python package manager that replaces pip, venv, poetry. It reminds me of `npm` or `cargo` if you are familiar from NodeJS or Rust. The benefits are fast installs, lockfiles, simple commands.

### Vast.ai

* cheap rented GPUs
* perfect for short-lived training

*Nb. if you want vast.ai [Secure Cloud](https://docs.vast.ai/documentation/reference/faq/security#what-is-secure-cloud) with fully trusted instances, make sure you filter for them.
Otherwise, be careful not to download secure files or secret keys onto your vast.ai instance.*

### VS Code Remote SSH
* Edit remote files like local
* Notebooks work normally
* Integrated terminals

## Create a new project on your local computer with `uv`

### Scaffold Project

[Install `uv`](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Go to your projects folder (for me, that's `~/github`):

```bash
cd ~/github
```

Initialize the uv project:

```bash
uv init --package ml-translate
```
You get:

```bash
ml-translate
├── pyproject.toml
├── README.md
└── src
    └── ml_translate
        └── __init__.py
```

*nb. git is initialized automatically by uv.*

### Create the environment (still on your local computer)

```bash
uv venv
```

Add runtime dependencies:

```bash
# as an example, I use numpy and pytorch
uv add numpy torch
```

Add dev dependencies:

```bash
uv add --dev ipykernel jupyter # you will need these to run jupyter notebooks
uv add --dev matplotlib pytest pytest-cov ruff # some other dev deps that I use
```

`uv` creates `.venv/` and `uv.lock` to make the environment full reproducible.

### Project layout
I keep things pretty simple:

```bash
ml-translate/
├── src/        # library code to use in notebooks
├── notebooks/  # orchestration of experiments
├── data/       # datasets for train/val/test
├── tests/      # tests for the src files
├── .venv/
└── pyproject.toml
```

### Local development in VS Code
Open the project in VS Code.
Select Python interpreter (on Mac I use `CMD+SHFT+P` to see menu).
Seleect `./.venv/bin/python`
It's important to set the same interpreter for both notebooks and the workspace so linting works correctly.

Now you can:

* develop normally
* run notebooks

Commit and push (you might need to setup your remote repo first):
```bash
git add -A
git commit -m "initial commit"
git push
```

## Vast.ai
### Setup account

Sign up at **vast.ai** and follow their [SSH docs](https://docs.vast.ai/documentation/instances/connect/ssh).

### Create an SSH key on your local computer:
```bash
ssh-keygen -t ed25519 -C "<your email>"
```
Save as `~/.ssh/vast_ed25519`

Copy your public key:

```bash
cat ~/.ssh/vast_ed25519.pub | pbcopy
```

Add it in the Vast.ai dashboard under **SSH keys**.

### Rent a GPU

Create an instance in the Vast UI.
When it starts, in the ssh key UI, you’ll see something like:

```bash
ssh -p 42917 root@77.xxx.xxx.xxx -L 8080:localhost:8080
```

Take note of the port and the hostname.

### Add SSH config

We can modify `~/.ssh/config` to get a easy integration with VSCode:

```bash
# I use neovim, but you can use any editor.
nvim ~/.ssh/config
```
Add the following, changing the hostname and port to your instance:
```bash
Host vast-ai
  HostName 77.xxx.xxx.xxx
  Port 42917
  User root
  IdentityFile ~/.ssh/vast_ed25519
  LocalForward 8080 localhost:8080
  AddKeysToAgent yes
  ForwardAgent yes
```

### Connect with VS Code Remote SSH

In VSCode, install the extension **Remote – SSH**.

Then connect to your remote instance:
```
Cmd/Ctrl + Shift + P
Remote-SSH: Connect to Host → vast-ai
```

### Clone and sync the project on the GPU

Inside the remote session, clone your repo and install dependencies:
```bash
git clone git@github.com:nofable/ml-translate.git
cd ml-translate
uv sync
```

In VSCode, open your folder in the explorer `/workspace/ml-translate`:

You now have your project up and running on the remote host, but it feels like it's local.

### Run notebooks on the GPU

Go to one of your notebooks and try to run it.
Install Python + Jupyter if prompted.

Set interpreter to `/workspace/ml-translate/.venv/bin/python`

Now all notebooks execute directly on the GPU, and the charts show in VSCode.

### git on remote host
#### Enable git on the remote
```bash
git config --global user.email "<email>"
git config --global user.name "<username>"
```

#### Secure push to origin
Because we enabled SSH agent forwarding in our `~/.ssh/config` file with `ForwardAgent yes`, we can push to github without exposing our ssh key to the remote host:
```bash
git push origin main
```

### Monitor GPU usage
To see the GPU utilization of my training, I used this command in tmux:
```bash
watch -n 1 nvidia-smi
```

It shows GPU utilization, memory usage & running processes so its super useful during training to see if you are using the GPU effectively.

### Shut down when finished
Once training is done, I save my results, including the charts.
Then **destroy the instance** in Vast UI.
Otherwise you keep getting billed.

## My typical workflow

### Local
* code
* test
* short train loop to sanity check
* commit
* push to github

### Remote GPU
* connect via VS Code
* `uv sync`
* run notebooks / training
* push results
* destroy instance

# Why I like this setup
- fast installs
- cheap GPUs
- zero cloud complexity
- reproducible environments
- notebooks “just work”
- only pay when needed

