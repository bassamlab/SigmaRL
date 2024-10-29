# SigmaRL: A Sample-Efficient and Generalizable Multi-Agent Reinforcement Learning Framework for Motion Planning
<!-- icons from https://simpleicons.org/ -->

- [SigmaRL: A Sample-Efficient and Generalizable Multi-Agent Reinforcement Learning Framework for Motion Planning](#sigmarl-a-sample-efficient-and-generalizable-multi-agent-reinforcement-learning-framework-for-motion-planning)
  - [Welcome to SigmaRL!](#welcome-to-sigmarl)
  - [Install](#install)
    - [1. Installing via pip](#1-installing-via-pip)
    - [2. Installing from Source](#2-installing-from-source)
  - [How to Use](#how-to-use)
    - [Training](#training)
    - [Testing](#testing)
  - [Customize Your Own Maps](#customize-your-own-maps)
  - [News](#news)
  - [Publications](#publications)
    - [1. SigmaRL](#1-sigmarl)
    - [2. XP-MARL](#2-xp-marl)
  - [TODOs](#todos)
  - [Acknowledgements](#acknowledgements)

> [!NOTE]
> Check out our recent work [XP-MARL](#2-xp-marl)! It is an open-source framework that augments MARL with au<ins>x</ins>iliary <ins>p</ins>rioritization to address *non-stationarity* in cooperative MARL.

## Welcome to SigmaRL!
This repository provides the full code of **SigmaRL**, a <ins>S</ins>ample eff<ins>i</ins>ciency and <ins>g</ins>eneralization <ins>m</ins>ulti-<ins>a</ins>gent <ins>R</ins>einforcement <ins>L</ins>earning (MARL) for motion planning of Connected and Automated Vehicles (CAVs).

SigmaRL is a decentralized MARL framework designed for motion planning of CAVs. We use <a href="https://github.com/proroklab/VectorizedMultiAgentSimulator" target="_blank">VMAS</a>, a vectorized differentiable simulator designed for efficient MARL benchmarking, as our simulator und customize our own RL environment. The first scenario in [Fig. 1](#) mirrors the real-world coditions of our Cyber-Physical Mobility Lab (<a href="https://cpm.embedded.rwth-aachen.de/" target="_blank">CPM Lab</a>). Besides, we also support maps handcrafted in <a href="https://josm.openstreetmap.de/" target="_blank">JOSM</a>, an open-source editor for OpenStreetMap. [Below](#customize-your-own-maps) you will find detailed guidance to create your **OWN** maps.

<table>
  <tr>
    <td>
      <a id="fig-scenario-cpm"></a>
      <figure>
        <img src="https://github.com/cas-lab-munich/assets/blob/main/sigmarl/media/cpm_entire.gif?raw=true" width="360" height="320" />
        <br>
        <figcaption>(a) CPM scenario.</figcaption>
      </figure>
    </td>
    <td>
      <a id="fig-scenario-intersection"></a>
      <figure>
        <img src="https://github.com/cas-lab-munich/assets/blob/main/sigmarl/media/intersection_2.gif?raw=true" height="320"/>
        <br>
        <figcaption>(b) Intersection scenario.</figcaption>
      </figure>
    </td>
  </tr>
  <tr>
    <td>
      <a id="fig-scenario-on-ramp"></a>
      <figure>
        <img src="https://github.com/cas-lab-munich/assets/blob/main/sigmarl/media/on_ramp_1.gif?raw=true" width="360"/>
        <br>
        <figcaption>(c) On-ramp scenario.</figcaption>
      </figure>
    </td>
    <td>
      <a id="fig-scenario-roundabout"></a>
      <figure>
        <img src="https://github.com/cas-lab-munich/assets/blob/main/sigmarl/media/roundabout_1.gif?raw=true" height="140"/>
        <br>
        <figcaption>(d) "Roundabout" scenario.</figcaption>
      </figure>
    </td>
  </tr>
</table>


## Install
Ensure you have Python 3.9 or 3.10 installed. Other versions may also work well. This repository works well on Windows, macOS, and Ubuntu. It's recommended to use a virtual environment (optional):
  ```bash
  python3 -m venv .sigmarl-env
  source .sigmarl-env/bin/activate
  ```
Two options are available to install `sigmarl`:
### 1. Installing via pip
You can use pip to install the latest release:
  ```bash
  pip install sigmarl
  ```

### 2. Installing from Source
If you want to install the current version, you can do:
- Clone the repository
  ```bash
  git clone https://github.com/cas-lab-munich/SigmaRL.git
  cd SigmaRL
  pip install -e .
  ```
- Verifying the Installation by first launching the Python 3 interpreter in terminal:
  ```bash
  python3
  ```
  Then run the following lines, which should show the version of the installed `sigmarl`:
  ```bash
  import sigmarl
  print(sigmarl.__version__)
  ```

## How to Use
### Training
Run `/main_training.py`. During training, all the intermediate models that have higher performance than the saved one will be automatically saved. You are also allowed to retrain or refine a trained model by setting the parameter `is_continue_train` in the `config.json` from the root directory file to `true`. The saved model will be loaded for a new training process.

`/sigmarl/scenarios/road_traffic.py` defines the RL environment, such as observation function and reward function. Besides, it provides an interactive interface, which also visualizes the environment. To open the interface, simply run this file. You can use `arrow keys` to control agents and use the `tab key` to switch between agents. Adjust the parameter `scenario_type` to choose a scenario. All available scenarios are listed in the variable `SCENARIOS` in `sigmarl/constants.py`. It is recommended to use the virtual visualization to check if the environment is as expected before training.
### Testing
After training, run `/main_testing.py` to test your model. You may need to adjust the parameter `path` therein to tell which folder the target model was saved.
*Note*: If the path to a saved model changes, you need to update the value of `where_to_save` in the corresponding JSON file as well.

## Customize Your Own Maps
We support maps customized in <a href="https://josm.openstreetmap.de/" target="_blank">JOSM</a>, an open-source editor for ​OpenStreetMap. Follow these steps:
- Install and open JOSM, click the green download button
- Zoom in and find an empty area (as empty as possible)
- Select the area by drawing a rectangle
- Click "Download"
- Now you will see a new window. Make sure there is no element. Otherwise, redo the above steps.
- Customize lanes. Note that all lanes you draw are considered center lines. You do not need to draw left and right boundaries, since they will be determined automatically later by our script with a given width.
- Save the osm file and store it at `sigmarl.assets/maps`. Give it a name.
- Go to `sigmarl/constants.py` and create a new dictionary for it. You should at least give the value for the key `map_path`, `lane_width`, and `scale`.
- Go to `sigmarl/parse_osm.py`. Adjust the parameters `scenario_type` and run it.

## News
- [2024-09-15] Check out our recent work [XP-MARL](#2-xp-marl), an open-source framework that augments MARL with au<ins>x</ins>iliary <ins>p</ins>rioritization to address *non-stationarity* in cooperative MARL!
- [2024-08-14] We support customized maps in OpenStreetMap now (see [here](#customize-your-own-maps))!
- [2024-07-10] Our [CPM Scenario](#fig-scenario-cpm) is now available as an MARL benchmark scenario in VMAS (see <a href="https://github.com/proroklab/VectorizedMultiAgentSimulator/releases/tag/1.4.2" target="_blank">here</a>)!
- [2024-07-10] Our work [SigmaRL](#1-sigmarl) was accepted by the 27th IEEE International Conference on Intelligent Transportation Systems (IEEE ITSC 2024)!

## Publications
We would be grateful if you would refer to the papers below if you find this repository helpful.


### 1. SigmaRL
<div>
Jianye Xu, Pan Hu, and Bassam Alrifaee, "SigmaRL: A Sample-Efficient and Generalizable Multi-Agent Reinforcement Learning Framework for Motion Planning," <i>2024 IEEE 27th International Conference on Intelligent Transportation Systems (ITSC), in press</i>, 2024.

<a href="https://doi.org/10.48550/arXiv.2408.07644" target="_blank"><img src="https://img.shields.io/badge/-Preprint-b31b1b?logo=arXiv"></a> <a href="https://youtu.be/tzaVjol4nhA" target="_blank"><img src="https://img.shields.io/badge/-Video-FF0000?logo=YouTube"></a> <a href="https://github.com/cas-lab-munich/SigmaRL/tree/1.2.0" target="_blank"><img src="https://img.shields.io/badge/-GitHub-181717?logo=GitHub"></a>
</div>

- **BibTeX**
  ```bibtex
  @inproceedings{xu2024sigmarl,
    title={{{SigmaRL}}: A Sample-Efficient and Generalizable Multi-Agent Reinforcement Learning Framework for Motion Planning},
    author={Xu, Jianye and Hu, Pan and Alrifaee, Bassam},
    booktitle={2024 IEEE 27th International Conference on Intelligent Transportation Systems (ITSC), in press},
    year={2024},
    organization={IEEE}
  }
  ```

- **Reproduce Experimental Results in the Paper:**

  - Git checkout to the corresponding tag using `git checkout 1.2.0`
  - Go to [this page](https://github.com/cas-lab-munich/assets/blob/main/sigmarl/checkpoints.zip) and download the zip file `checkpoints.zip`. Unzip it, copy and paste the whole folder `checkpoints` to the **root** of this repository.
  - Run `sigmarl/evaluation_itsc24.py`.

  You can also run `/testing_mappo_cavs.py` to intuitively evaluate the trained models. Adjust the parameter `path` therein to specify which folder the target model was saved.
  *Note*: The evaluation results you get may slightly deviate from the paper since we carefully improved computation of the performance metrics.


### 2. XP-MARL
<div>
Jianye Xu, Omar Sobhy, and Bassam Alrifaee, "XP-MARL: Auxiliary Prioritization in Multi-Agent Reinforcement Learning to Address Non-Stationarity," <i>arXiv preprint arXiv:2409.11852</i>, 2024.

<a href="https://doi.org/10.48550/arXiv.2409.11852" target="_blank"><img src="https://img.shields.io/badge/-Preprint-b31b1b?logo=arXiv"></a> <a href="https://youtu.be/GEhjRKY2fTU" target="_blank"><img src="https://img.shields.io/badge/-Video-FF0000?logo=YouTube"></a> <a href="https://github.com/cas-lab-munich/SigmaRL/tree/1.2.0" target="_blank"><img src="https://img.shields.io/badge/-GitHub-181717?logo=GitHub"></a>
</div>

- **BibTeX**
  ```bibtex
  @article{xu2024xp,
    title={{{XP-MARL}}: Auxiliary Prioritization in Multi-Agent Reinforcement Learning to Address Non-Stationarity},
    author={Xu, Jianye and Sobhy, Omar and Alrifaee, Bassam},
    journal={arXiv preprint arXiv:2409.11852},
    year={2024},
  }
  ```

- **Reproduce Experimental Results in the Paper:**

  - Git checkout to the corresponding tag using `git checkout 1.2.0`
  - Go to [this page](https://github.com/cas-lab-munich/assets/blob/main/sigmarl/checkpoints.zip) and download the zip file `checkpoints.zip`. Unzip it, copy and paste the whole folder `checkpoints` to the **root** of this repository.
  - Run `sigmarl/evaluation_icra25.py`.

  You can also run `/testing_mappo_cavs.py` to intuitively evaluate the trained models. Adjust the parameter `path` therein to specify which folder the target model was saved.

## TODOs
- Effective observation design
  - [ ] Image-based representation of observations
  - [ ] Historic observations
  - [ ] Attention mechanism
- Improve safety
  - [ ] Integrating Control Barrier Functions (CBFs)
  - [ ] Integrating Model Predictive Control (MPC)
- Address non-stationarity
  - [x] Integrating prioritization (see the XP-MARL paper [here](#2-xp-marl))
- Misc
  - [x] OpenStreetMap support (see guidance [here](#customize-your-own-maps))
  - [x] Contribute our [CPM scenario](#fig-scenario-cpm) as an MARL benchmark scenario in VMAS (see news <a href="https://github.com/proroklab/VectorizedMultiAgentSimulator/releases/tag/1.4.2" target="_blank">here</a>)

## Acknowledgements
This research was supported by the Bundesministerium für Digitales und Verkehr (German Federal Ministry for Digital and Transport) within the project "Harmonizing Mobility" (grant number 19FS2035A).
