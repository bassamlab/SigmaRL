# SigmaRL: A Sample-Efficient and Generalizable Multi-Agent Reinforcement Learning Framework for Motion Planning
<!-- icons from https://simpleicons.org/ -->
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/bassamlab/SigmaRL/blob/main/LICENSE.txt)
[![arXiv](https://img.shields.io/badge/arXiv-2408.07644-b31b1b.svg)](https://doi.org/10.48550/arXiv.2408.07644)
[![arXiv](https://img.shields.io/badge/arXiv-2409.11852-b31b1b.svg)](https://doi.org/10.48550/arXiv.2409.11852)
[![arXiv](https://img.shields.io/badge/arXiv-2411.08999-b31b1b.svg)](https://doi.org/10.48550/arXiv.2411.08999)

- [SigmaRL: A Sample-Efficient and Generalizable Multi-Agent Reinforcement Learning Framework for Motion Planning](#sigmarl-a-sample-efficient-and-generalizable-multi-agent-reinforcement-learning-framework-for-motion-planning)
  - [Welcome to SigmaRL!](#welcome-to-sigmarl)
  - [Install](#install)
  - [How to Use](#how-to-use)
    - [Training](#training)
    - [Testing](#testing)
  - [Customize Your Own Maps](#customize-your-own-maps)
  - [News](#news)
  - [Papers](#papers)
    - [1. SigmaRL](#1-sigmarl)
    - [2. XP-MARL](#2-xp-marl)
    - [3. CBF-MARL](#3-cbf-marl)
  - [TODOs](#todos)
  - [Acknowledgments](#acknowledgments)

> [!NOTE]
> - Check out our recent work [CBF-MARL](#3-cbf-marl)! It uses a learning-based, *less conservative* distance metric to categorize safety margins between agents and integrates it into Control Barrier Functions (CBFs) to guarantee *safety* in MARL.
> - Check out our recent work [XP-MARL](#2-xp-marl)! It augments MARL with learning-based au<ins>x</ins>iliary <ins>p</ins>rioritization to address *non-stationarity*.

## Welcome to SigmaRL!
This repository provides the full code of **SigmaRL**, a <ins>S</ins>ample eff<ins>i</ins>ciency and <ins>g</ins>eneralization <ins>m</ins>ulti-<ins>a</ins>gent <ins>R</ins>einforcement <ins>L</ins>earning (MARL) for motion planning of Connected and Automated Vehicles (CAVs).

SigmaRL is a decentralized MARL framework designed for motion planning of CAVs. We use <a href="https://github.com/proroklab/VectorizedMultiAgentSimulator" target="_blank">VMAS</a>, a vectorized differentiable simulator designed for efficient MARL benchmarking, as our simulator and customize our own RL environment. The first scenario in [Fig. 1](#) mirrors the real-world conditions of our Cyber-Physical Mobility Lab (<a href="https://cpm.embedded.rwth-aachen.de/" target="_blank">CPM Lab</a>). We also support maps handcrafted in <a href="https://josm.openstreetmap.de/" target="_blank">JOSM</a>, an open-source editor for OpenStreetMap. [Below](#customize-your-own-maps) you will find detailed guidance to create your **OWN** maps.

<figure>
  <table>
    <tr>
      <td>
        <a id="fig-scenario-cpm"></a>
        <figure>
          <img src="https://github.com/bassamlab/assets/blob/main/sigmarl/media/generalizable_cpm_entire.gif?raw=true" width="360" />
          <br>
          <figcaption>(a) CPM scenario.</figcaption>
        </figure>
      </td>
      <td>
        <a id="fig-scenario-intersection"></a>
        <figure>
          <img src="https://github.com/bassamlab/assets/blob/main/sigmarl/media/generalizable_intersection.gif?raw=true" width="200"/>
          <br>
          <figcaption>(b) Intersection scenario.</figcaption>
        </figure>
      </td>
    </tr>
    <tr>
      <td>
        <a id="fig-scenario-on-ramp"></a>
        <figure>
          <img src="https://github.com/bassamlab/assets/blob/main/sigmarl/media/generalizable_on_ramp.gif?raw=true" width="360"/>
          <br>
          <figcaption>(c) On-ramp scenario.</figcaption>
        </figure>
      </td>
      <td>
        <a id="fig-scenario-roundabout"></a>
        <figure>
          <img src="https://github.com/bassamlab/assets/blob/main/sigmarl/media/generalizable_roundabout.gif?raw=true" width="300"/>
          <br>
          <figcaption>(d) "Roundabout" scenario.</figcaption>
        </figure>
      </td>
    </tr>
  </table>
  <!-- <figcaption>Figure 1.</figcaption> -->
</figure>

Figure 1: Demonstrating the *generalization* of SigmaRL (speed x2). Only the intersection part of the CPM scenario (the middle part in Fig. 1(a)) is used for training. All other scenarios are completely unseen. See our [SigmaRL paper](#1-sigmarl) for more details.

<figure>
  <img src="https://github.com/bassamlab/assets/blob/main/sigmarl/media/xp-marl.gif?raw=true" width="400" />
  <!-- <figcaption>Figure 2:.</figcaption> -->
</figure>

Figure 2: We use an auxiliary MARL to learn dynamic priority assignments to address *non-stationarity*. Higher-priority agents communicate their actions (depicted by the colored lines) to lower-priority agents to stabilize the environment. See our [XP-MARL paper](#2-xp-marl) for more details.

<figure>
  <table>
    <tr>
      <td>
        <a id="fig-scenario-cpm"></a>
        <figure>
          <img src="https://github.com/bassamlab/assets/blob/main/sigmarl/media/eva_cbf_overtaking_c2c.gif?raw=true" width="400" />
          <br>
          <figcaption>(a) Overtaking scenario with Center-to-Center (C2C)-based safety margin (traditional).</figcaption>
        </figure>
      </td>
      <td>
        <a id="fig-scenario-intersection"></a>
        <figure>
          <img src="https://github.com/bassamlab/assets/blob/main/sigmarl/media/eva_cbf_overtaking_mtv.gif?raw=true" width="400"/>
          <br>
          <figcaption>(b) Overtaking scenario with Minimum Translation Vector (MTV)-based safety margin (<b>ours</b>).</figcaption>
        </figure>
      </td>
    </tr>
    <tr>
      <td>
        <a id="fig-scenario-on-ramp"></a>
        <figure>
          <img src="https://github.com/bassamlab/assets/blob/main/sigmarl/media/eva_cbf_bypassing_c2c.gif?raw=true" width="400"/>
          <br>
          <figcaption>(c) Bypassing scenario with C2C-based safety margin (traditional).</figcaption>
        </figure>
      </td>
      <td>
        <a id="fig-scenario-roundabout"></a>
        <figure>
          <img src="https://github.com/bassamlab/assets/blob/main/sigmarl/media/eva_cbf_bypassing_mtv.gif?raw=true" width="400"/>
          <br>
          <figcaption>(d) Bypassing scenario with MTV-based safety margin (<b>ours</b>).</figcaption>
        </figure>
      </td>
    </tr>
  </table>
  <!-- <figcaption>Figure 1.</figcaption> -->
</figure>

Figure 3: Demonstrating the safety and reduced conservatism of our MTV-based safety margin. In the overtaking scenario, while the traditional approach fails to overtake due to excessive conservatism (see (a)), ours succeeds (see (b)). Note that in the overtaking scenario, the slow-moving vehicle $j$ purposely obstructs vehicle $i$ three times to prevent it from overtaking. In the bypassing scenario, while the traditional approach requires a large lateral space due to excessive conservatism (see (c)), ours requires a smaller one (see (d)). See our [CBF-MARL paper](#3-cbf-marl) for more details.

## Install
Currently, `SigmaRL` supports Python versions 3.9 and 3.10 and is also OS independent (Windows/macOS/Linux). It's recommended to use a virtual environment. For example, if you are using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html):
  ```bash
  conda create -n sigmarl python=3.10
  conda activate sigmarl
  ```
We recommend installing `sigmarl` from source:
- Clone the repository
  ```bash
  git clone https://github.com/bassamlab/SigmaRL.git
  cd SigmaRL
  pip install -e .
  ```
- (Optional) Verifying the Installation by first launching your Python interpreter in terminal:
  ```bash
  python
  ```
  Then run the following lines, which should show the version of the installed `sigmarl`:
  ```bash
  import sigmarl
  print(sigmarl.__version__)
  ```

## How to Use
### Training
Run `main_training.py`. During training, all the intermediate models that have higher performance than the saved one will be automatically saved. You are also allowed to retrain or refine a trained model by setting the parameter `is_continue_train` in the file `sigmarl/config.json` to `true`. The saved model will be loaded for a new training process.

`sigmarl/scenarios/road_traffic.py` defines the RL environment, such as the observation function and reward function. Besides, it provides an interactive interface, which also visualizes the environment. To open the interface, simply run this file. You can use `arrow keys` to control agents and use the `tab key` to switch between agents. Adjust the parameter `scenario_type` to choose a scenario. All available scenarios are listed in the variable `SCENARIOS` in `sigmarl/constants.py`. Before training, it is recommended to use the interactive interface to check if the environment is as expected.
### Testing
After training, run `main_testing.py` to test your model. You may need to adjust the parameter `path` therein to tell which folder the target model was saved.
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
- [2024-11-15] Check out our recent work [CBF-MARL](#3-cbf-marl)! It uses a learning-based, *less conservative* distance metric to quantify safety margins between agents and integrates it into Control Barrier Functions (CBFs) to guarantee *safety* in MARL.
- [2024-09-15] Check out our recent work [XP-MARL](#2-xp-marl)! It augments MARL with learning-based au<ins>x</ins>iliary <ins>p</ins>rioritization to address *non-stationarity*.
- [2024-08-14] We support customized maps in OpenStreetMap now (see [here](#customize-your-own-maps))!
- [2024-07-10] Our [CPM Scenario](#fig-scenario-cpm) is now available as an MARL benchmark scenario in VMAS (see <a href="https://github.com/proroklab/VectorizedMultiAgentSimulator/releases/tag/1.4.2" target="_blank">here</a>)!
- [2024-07-10] Our work [SigmaRL](#1-sigmarl) was accepted by the 27th IEEE International Conference on Intelligent Transportation Systems (IEEE ITSC 2024)!

## Papers
We would be grateful if you would refer to the papers below if you find this repository helpful.


### 1. SigmaRL
<div>
Jianye Xu, Pan Hu, and Bassam Alrifaee, "SigmaRL: A Sample-Efficient and Generalizable Multi-Agent Reinforcement Learning Framework for Motion Planning," <i>2024 IEEE 27th International Conference on Intelligent Transportation Systems (ITSC), in press</i>, 2024.

<a href="https://doi.org/10.48550/arXiv.2408.07644" target="_blank"><img src="https://img.shields.io/badge/-Preprint-b31b1b?logo=arXiv"></a> <a href="https://youtu.be/tzaVjol4nhA" target="_blank"><img src="https://img.shields.io/badge/-Video-FF0000?logo=YouTube"></a> <a href="https://github.com/bassamlab/SigmaRL/tree/1.2.0" target="_blank"><img src="https://img.shields.io/badge/-GitHub-181717?logo=GitHub"></a>
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
  - Go to [this page](https://github.com/bassamlab/assets/blob/main/sigmarl/checkpoints/itsc24.zip) and download the zip file `itsc24.zip`. Unzip it, copy and paste the whole folder to the `checkpoints` folder at the **root** of this repository. The structure should be like this: `root/checkpoints/itsc24/`.
  - Run `sigmarl/evaluation_itsc24.py`.

  You can also run `testing_mappo_cavs.py` to intuitively evaluate the trained models. Adjust the parameter `path` therein to specify which folder the target model was saved.
  *Note*: The evaluation results you get may deviate from the paper since we have meticulously adjusted the performance metrics.


### 2. XP-MARL
<div>
Jianye Xu, Omar Sobhy, and Bassam Alrifaee, "XP-MARL: Auxiliary Prioritization in Multi-Agent Reinforcement Learning to Address Non-Stationarity," <i>arXiv preprint arXiv:2409.11852</i>, 2024.

<a href="https://doi.org/10.48550/arXiv.2409.11852" target="_blank"><img src="https://img.shields.io/badge/-Preprint-b31b1b?logo=arXiv"></a> <a href="https://youtu.be/GEhjRKY2fTU" target="_blank"><img src="https://img.shields.io/badge/-Video-FF0000?logo=YouTube"></a> <a href="https://github.com/bassamlab/SigmaRL/tree/1.2.0" target="_blank"><img src="https://img.shields.io/badge/-GitHub-181717?logo=GitHub"></a>
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
  - Go to [this page](https://github.com/bassamlab/assets/blob/main/sigmarl/checkpoints/icra25.zip) and download the zip file `icra25.zip`. Unzip it, copy and paste the whole folder to the `checkpoints` folder at the **root** of this repository. The structure should be like this: `root/checkpoints/icra25/`.
  - Run `sigmarl/evaluation_icra25.py`.

  You can also run `testing_mappo_cavs.py` to intuitively evaluate the trained models. Adjust the parameter `path` therein to specify which folder the target model was saved.

### 3. CBF-MARL
<div>
Jianye Xu and Bassam Alrifaee, "Learning-Based Control Barrier Function with Provably Safe Guarantees: Reducing Conservatism with Heading-Aware Safety Margin," <i>arXiv preprint arXiv:2411.08999</i>, 2024.

<a href="https://doi.org/10.48550/arXiv.2411.08999" target="_blank"><img src="https://img.shields.io/badge/-Preprint-b31b1b?logo=arXiv"></a>
</div>

- **BibTeX**
  ```bibtex
  @article{xu2024learning,
    title={Learning-Based Control Barrier Function with Provably Safe Guarantees: Reducing Conservatism with Heading-Aware Safety Margin},
    author={Xu, Jianye and Alrifaee, Bassam},
    journal={arXiv preprint arXiv:2411.08999},
    year={2024},
  }
  ```

- **Reproduce Experimental Results in the Paper:**

  <!-- - Git checkout to the corresponding tag using `git checkout 1.3.0` TODO -->
  - Go to [this page](https://github.com/bassamlab/assets/blob/main/sigmarl/checkpoints/ecc25.zip) and download the zip file `ecc25.zip`. Unzip it, copy and paste the whole folder to the `checkpoints` folder at the **root** of this repository. The structure should be like this: `root/checkpoints/ecc25/`.
  - Run `sigmarl/evaluation_ecc25.py`.

## TODOs
- Effective observation design
  - [ ] Image-based representation of observations
  - [ ] Historic observations
  - [ ] Attention mechanism
- Improve safety
  - [ ] Integrating Control Barrier Functions (CBFs)
    - [x] Proof of concept with two agents (see the CBF-MARL paper [here](#3-cbf-marl))
  - [ ] Integrating Model Predictive Control (MPC)
- Address non-stationarity
  - [x] Integrating prioritization (see the XP-MARL paper [here](#2-xp-marl))
- Misc
  - [x] OpenStreetMap support (see guidance [here](#customize-your-own-maps))
  - [x] Contribute our [CPM scenario](#fig-scenario-cpm) as an MARL benchmark scenario in VMAS (see news <a href="https://github.com/proroklab/VectorizedMultiAgentSimulator/releases/tag/1.4.2" target="_blank">here</a>)
  - [ ] Update to the latest versions of Torch, TorchRL, and VMAS
  - [ ] Support Python 3.11+

## Acknowledgments
This research was supported by the Bundesministerium für Digitales und Verkehr (German Federal Ministry for Digital and Transport) within the project "Harmonizing Mobility" (grant number 19FS2035A).
