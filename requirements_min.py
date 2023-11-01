a = """
absl-py==2.0.0
asttokens==2.4.0
backcall==0.2.0
beartype==0.16.3
certifi==2023.7.22
charset-normalizer==3.3.0
chex==0.1.83
contourpy==1.1.1
cycler==0.12.1
decorator==5.1.1
einops==0.7.0
etils==1.5.1
executing==2.0.0
flax==0.7.4
fonttools==4.43.1
fsspec==2023.9.2
idna==3.4
importlib-resources==6.1.0
ipdb==0.13.13
ipython==8.16.1
jax==0.4.17
jaxlib==0.4.17
jaxtyping==0.2.23
jedi==0.19.1
kiwisolver==1.4.5
libtpu-nightly==0.1.dev20231003
markdown-it-py==3.0.0
-e git+https://github.com/salamanderxing/mate@cf150300464b8e9669449c765a898827ec936413#egg=mate
matplotlib==3.8.0
matplotlib-inline==0.1.6
mdurl==0.1.2
ml-dtypes==0.3.1
msgpack==1.0.7
nest-asyncio==1.5.8
networkx==3.1
numpy==1.26.1
opt-einsum==3.3.0
optax==0.1.7
orbax-checkpoint==0.4.1
packaging==23.2
pandas==2.1.1
parso==0.8.3
pexpect==4.8.0
pickleshare==0.7.5
Pillow==10.0.1
prompt-toolkit==3.0.39
protobuf==4.24.4
ptyprocess==0.7.0
pure-eval==0.2.2
pyaml==23.9.7
Pygments==2.16.1
pyparsing==3.1.1
python-dateutil==2.8.2
pytz==2023.3.post1
PyYAML==6.0.1
rdkit @ file:///home/bluesk/Documents/rdkit-2023.3.3-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl#sha256=89cf60a2369bea768d50e0d3bcb8219993c105c999631daea492f86cf6c6cba6
requests==2.31.0
rich==13.6.0
scipy==1.11.3
six==1.16.0
stack-data==0.6.3
tensorstore @ file:///tmp/tensorstore-0.1.45-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl#sha256=ca212d127fcc4debb9f6b4274d584fe7724b2a349ca9444258a4127878dc3033
toolz==0.12.0
tqdm==4.66.1
traitlets==5.11.2
typeguard==2.13.3
typing_extensions==4.8.0
tzdata==2023.3
urllib3==2.0.6
wcwidth==0.2.8
zipp==3.17.0
""".splitlines()
a = [i.strip().split("=")[0] for i in a if "=" in i]


with open("requirements_min.txt", "w") as f:
    f.write("\n".join(a))
