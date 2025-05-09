# SigmaRL: A Sample-Efficient and Generalizable Multi-Agent Reinforcement Learning Framework for Motion Planning
<!-- icons from https://simpleicons.org/ -->
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/bassamlab/SigmaRL/blob/main/LICENSE.txt)
[![arXiv](https://img.shields.io/badge/arXiv-2408.07644-b31b1b.svg)](https://doi.org/10.48550/arXiv.2408.07644)
[![arXiv](https://img.shields.io/badge/arXiv-2409.11852-b31b1b.svg)](https://doi.org/10.48550/arXiv.2409.11852)
[![arXiv](https://img.shields.io/badge/arXiv-2411.08999-b31b1b.svg)](https://doi.org/10.48550/arXiv.2411.08999)
[![arXiv](https://img.shields.io/badge/arXiv-2503.15014-b31b1b.svg)](https://doi.org/10.48550/arXiv.2503.15014)
[![arXiv](https://img.shields.io/badge/arXiv-2503.15014-b31b1b.svg)](https://doi.org/10.48550/arXiv.2505.02395)

- [SigmaRL: A Sample-Efficient and Generalizable Multi-Agent Reinforcement Learning Framework for Motion Planning](#sigmarl-a-sample-efficient-and-generalizable-multi-agent-reinforcement-learning-framework-for-motion-planning)
  - [Welcome to SigmaRL!](#welcome-to-sigmarl)
  - [Install](#install)
  - [How to Use](#how-to-use)
    - [Training](#training)
    - [Testing](#testing)
  - [Customize Your Own Maps](#customize-your-own-maps)
  - [Papers](#papers)
    - [1. SigmaRL](#1-sigmarl)
    - [2. XP-MARL](#2-xp-marl)
    - [3. MTV-Based CBF](#3-mtv-based-cbf)
    - [4. Truncated Taylor CBF (TTCBF)](#4-truncated-taylor-cbf-ttcbf)
    - [5. CBF-Based Safety Filter](#5-cbf-based-safety-filter)
  - [TODOs](#todos)
  - [Acknowledgments](#acknowledgments)

> [!NOTE]
> - Check out our recent work [CBF-Based Safety Filter](#5-cbf-based-safety-filter)! It proposes a real-time CBF-based safety filter for safety verification of learning-based motion planning with road boundary constraints (see also [Fig. 5](#fig-safety-filter)).
> - Check out our recent work [Truncated Taylor CBF](#4-truncated-taylor-cbf-ttcbf)! It proposes a new notion of high-order CBFs termed Truncated Taylor CBF (TTCBF). TTCBF can handle constraints with arbitrary relative degrees while using only one design parameter to facilitate control design (see also [Fig. 4](#fig-ttcbf)).
<!-- > - Check out our recent work [MTV-Based CBF](#3-mtv-based-cbf)! It uses a learning-based, *less conservative* distance metric to categorize safety margins between agents and integrates it into Control Barrier Functions (CBFs) to guarantee *safety* in MARL. -->
<!-- > - Check out our recent work [XP-MARL](#2-xp-marl)! It augments MARL with learning-based au<ins>x</ins>iliary <ins>p</ins>rioritization to address *non-stationarity*. -->

## Welcome to SigmaRL!
This repository provides the full code of **SigmaRL**, a <ins>S</ins>ample eff<ins>i</ins>ciency and <ins>g</ins>eneralization <ins>m</ins>ulti-<ins>a</ins>gent <ins>R</ins>einforcement <ins>L</ins>earning (MARL) for motion planning of Connected and Automated Vehicles (CAVs).

SigmaRL is a decentralized MARL framework designed for motion planning of CAVs. We use <a href="https://github.com/proroklab/VectorizedMultiAgentSimulator" target="_blank">VMAS</a>, a vectorized differentiable simulator designed for efficient MARL benchmarking, as our simulator and customize our own RL environment. The first scenario in [Fig. 1](#) mirrors the real-world conditions of our Cyber-Physical Mobility Lab (<a href="https://cpm.embedded.rwth-aachen.de/" target="_blank">CPM Lab</a>). We also support maps handcrafted in <a href="https://josm.openstreetmap.de/" target="_blank">JOSM</a>, an open-source editor for OpenStreetMap. [Below](#customize-your-own-maps) you will find detailed guidance to create your **OWN** maps.

<a id="fig-generalization"></a>
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

<a id="fig-xp-marl"></a>
<figure>
  <img src="https://github.com/bassamlab/assets/blob/main/sigmarl/media/xp-marl.gif?raw=true" width="400" />
  <!-- <figcaption>Figure 2:.</figcaption> -->
</figure>

Figure 2: We use an auxiliary MARL to learn dynamic priority assignments to address *non-stationarity*. Higher-priority agents communicate their actions (depicted by the colored lines) to lower-priority agents to stabilize the environment. See our [XP-MARL paper](#2-xp-marl) for more details.

<a id="fig-mtv-based-cbf"></a>
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
</figure>

Figure 3: Demonstrating the safety and reduced conservatism of our MTV-based safety margin. In the overtaking scenario, while the traditional approach fails to overtake due to excessive conservatism (see (a)), ours succeeds (see (b)). Note that in the overtaking scenario, the slow-moving vehicle $j$ purposely obstructs vehicle $i$ three times to prevent it from overtaking. In the bypassing scenario, while the traditional approach requires a large lateral space due to excessive conservatism (see (c)), ours requires a smaller one (see (d)). See our [MTV-Based CBF paper](#3-mtv-based-cbf) for more details.

<a id="fig-ttcbf"></a>
<figure>
  <table>
    <tr>
      <td>
        <a id="fig-ttcbf-hocbf"></a>
        <figure>
          <img src="https://github.com/bassamlab/assets/blob/main/sigmarl/media/ttcbf_experiment_hocbf.jpg?raw=true" width="800" />
          <br>
          <figcaption>(a) The standard HOCBF approach requires tuning two parameters (lambda_1 and lambda_2).</figcaption>
        </figure>
      </td>
    </tr>
    <tr>
      <td>
        <a id="fig-ttcbf-taylor"></a>
        <figure>
          <img src="https://github.com/bassamlab/assets/blob/main/sigmarl/media/ttcbf_experiment_taylor.jpg?raw=true" width="800"/>
          <br>
          <figcaption>(b) Our TTCBF HOCBF approach requires tuning only one parameter (lambda_1).</figcaption>
        </figure>
      </td>
    </tr>
  </table>
</figure>

Figure 4: Our TTCBF approach reduces the number of parameters to tune when handling constraints with high relative degrees. See our [TTCBF paper](#4-truncated-taylor-cbf-ttcbf) for more details.

<a id="fig-safety-filter"></a>
<figure>
  <table>
    <tr>
      <td>
        <a id="fig-ttcbf-hocbf"></a>
        <figure>
          <img src="https://github.com/bassamlab/assets/blob/main/sigmarl/media/safety_filter_without.gif?raw=true" width="400" />
          <br>
          <figcaption>(a) An undertrained RL policy without our safety filter often caused collisions with road boundaries.</figcaption>
        </figure>
      </td>
      <td>
        <a id="fig-ttcbf-taylor"></a>
        <figure>
          <img src="https://github.com/bassamlab/assets/blob/main/sigmarl/media/safety_filter_with.gif?raw=true" width="400"/>
          <br>
          <figcaption>(b) Our safety filter successfully avoided all collisions caused by the undertrained RL policy.</figcaption>
        </figure>
      </td>
    </tr>
  </table>
</figure>

Figure 5: Demonstration of our safety filter for safety verification of an undertrained RL policy. See our [CBF-Based Safety Filter Paper](#5-cbf-based-safety-filter) for more details.

## Install
SigmaRL supports Python versions from 3.9 to 3.12 and is also OS independent (Windows/macOS/Linux). It's recommended to use a virtual environment. For example, if you are using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html):
  ```bash
  conda create -n sigmarl python=3.12
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
We support maps customized in <a href="https://josm.openstreetmap.de/" target="_blank">JOSM</a>, an open-source editor for ​OpenStreetMap. Follow these steps (video tutorial available <a href="https://github.com/bassamlab/assets/blob/main/sigmarl/media/custom_map_tutorial.mp4" target="_blank">here</a>):
- Install JOSM from the website given above.
- To get an empty map that can be customized, do the following:
  - Open JOSM and click the green download button
  - Zoom in and choose an arbitrary place on the map by drawing a rectangle. The area should be as empty as possible.
  - Clicking "Download" will open a new window. There should be the notification that no data could get found, otherwise redo choosing the area.
- Customize the map by drawing lines. Note that all lanes you draw are considered center lines. You do not need to draw left and right boundaries, since they will be determined automatically later by our script with a given width. The distance between the nodes of a lane should be approximatly 0.1 meters. You can find useful hints and commands for customizing the map at <a href="https://josm.openstreetmap.de/wiki/Help/Action/Select#Scale" target="_blank">Actions</a> and <a href="https://josm.openstreetmap.de/wiki/Help/Menu/Tools" target="_blank">Tools</a>.
- Give each lane the key "lanes" and an unique value.
- Save the resulting .osm file and store it at `assets/maps`. Give it a name.
- Go to `utilities/constants.py` and create a new entry in the dictionary "SCENARIOS" for it. The key of the entry is the name of the map and the value is a dictionary, for which you should at least give the value for the key `map_path`, `lane_width`, and `scale`. Also you should provide a list for `reference_paths_ids` (which paths exist?) and a dictionary for `neighboring_lanelet_ids` (which lanes are adjacent?).
- Go to `utilities/parse_osm.py`. Adjust the parameters `scenario_type` and run it.

<img src="https://github.com/bassamlab/assets/blob/main/sigmarl/media/maps_overview.jpg?raw=true" alt="Overview Map" style="max-width:800px; width:100%;">
<a id="fig-maps"></a>
Figure 6: Overview of currently available maps.


## Papers
If you use this repository, please consider to cite our papers.

### 1. SigmaRL
<div>
Jianye Xu, Pan Hu, and Bassam Alrifaee, "SigmaRL: A Sample-Efficient and Generalizable Multi-Agent Reinforcement Learning Framework for Motion Planning," <i>2024 IEEE 27th International Conference on Intelligent Transportation Systems (ITSC), Edmonton, AB, Canada, 2024, pp. 768-775, doi: 10.1109/ITSC58415.2024.10919918</i>.

<a href="https://doi.org/10.48550/arXiv.2408.07644" target="_blank"><img src="https://img.shields.io/badge/-Preprint-b31b1b?logo=arXiv"></a> <a href="https://youtu.be/tzaVjol4nhA" target="_blank"><img src="https://img.shields.io/badge/-Video-FF0000?logo=YouTube"></a> [![Jump to Fig. 1](https://img.shields.io/badge/Jump%20to-Fig.%201-blue)](#fig-generalization) <a href="https://github.com/bassamlab/SigmaRL/tree/1.2.0" target="_blank"><img src="https://img.shields.io/badge/-GitHub-181717?logo=GitHub"></a>
</div>

- **BibTeX**
  ```bibtex
  @inproceedings{xu2024sigmarl,
    title = {SigmaRL: A Sample-Efficient and Generalizable Multi-Agent Reinforcement Learning Framework for Motion Planning},
    booktitle = {2024 IEEE 27th International Conference on Intelligent Transportation Systems (ITSC)},
    author = {Xu, Jianye and Hu, Pan and Alrifaee, Bassam},
    year = {2024},
    pages = {768--775},
    issn = {2153-0017},
    doi = {10.1109/ITSC58415.2024.10919918}
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

<a href="https://doi.org/10.48550/arXiv.2409.11852" target="_blank"><img src="https://img.shields.io/badge/-Preprint-b31b1b?logo=arXiv"></a> <a href="https://youtu.be/GEhjRKY2fTU" target="_blank"><img src="https://img.shields.io/badge/-Video-FF0000?logo=YouTube"></a> [![Jump to Fig. 2](https://img.shields.io/badge/Jump%20to-Fig.%202-blue)](#fig-xp-marl) <a href="https://github.com/bassamlab/SigmaRL/tree/1.2.0" target="_blank"><img src="https://img.shields.io/badge/-GitHub-181717?logo=GitHub"></a>
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

### 3. MTV-Based CBF
<div>
Jianye Xu and Bassam Alrifaee, "Learning-Based Control Barrier Function with Provably Safe Guarantees: Reducing Conservatism with Heading-Aware Safety Margin," <i>In European Control Conference (ECC), in press</i>, 2024.

<a href="https://doi.org/10.48550/arXiv.2411.08999" target="_blank"><img src="https://img.shields.io/badge/-Preprint-b31b1b?logo=arXiv"> </a>[![Jump to Fig. 3](https://img.shields.io/badge/Jump%20to-Fig.%203-blue)](#fig-mtv-based-cbf)
</div>

- **BibTeX**
  ```bibtex
  @inproceedings{xu2024learningbased,
    title = {Learning-Based Control Barrier Function with Provably Safe Guarantees: Reducing Conservatism with Heading-Aware Safety Margin},
    shorttitle = {Learning-Based Control Barrier Function with Provably Safe Guarantees},
    booktitle = {European Control Conference (ECC), in Press},
    author = {Xu, Jianye and Alrifaee, Bassam},
    year = {2025},
  }
  ```

- **Reproduce Experimental Results in the Paper:**

  <!-- - Git checkout to the corresponding tag using `git checkout 1.3.0` TODO -->
  - Go to [this page](https://github.com/bassamlab/assets/blob/main/sigmarl/checkpoints/ecc25.zip) and download the zip file `ecc25.zip`. Unzip it, copy and paste the whole folder to the `checkpoints` folder at the **root** of this repository. The structure should be like this: `root/checkpoints/ecc25/`.
  - Run `sigmarl/evaluation_ecc25.py`.

### 4. Truncated Taylor CBF (TTCBF)
<div>
Jianye Xu and Bassam Alrifaee, "High-Order Control Barrier Functions: Insights and a Truncated Taylor-Based Formulation," <i>arXiv preprint arXiv:2503.15014</i>, 2025.

<a href="https://doi.org/10.48550/arXiv.2503.15014" target="_blank"><img src="https://img.shields.io/badge/-Preprint-b31b1b?logo=arXiv"></a> [![Jump to Fig. 4](https://img.shields.io/badge/Jump%20to-Fig.%204-blue)](#fig-ttcbf) <a href="https://github.com/bassamlab/SigmaRL/tree/1.3.0" target="_blank"><img src="https://img.shields.io/badge/-GitHub-181717?logo=GitHub"></a>
</div>

- **BibTeX**
  ```bibtex
  @article{xu2025highorder,
    title = {High-Order Control Barrier Functions: Insights and a Truncated Taylor-Based Formulation},
    author = {Xu, Jianye and Alrifaee, Bassam},
    journal = {arXiv preprint arXiv:2503.15014},
    year = {2025},
  }
  ```

- **Reproduce Experimental Results in the Paper:**
  - Git checkout to the corresponding tag using `git checkout 1.3.0`
  - Run `sigmarl/hocbf_taylor.py`.

### 5. CBF-Based Safety Filter
<div>
Jianye Xu, Chang Che, and Bassam Alrifaee, "A Real-Time Control Barrier Function-Based Safety Filter for Motion Planning with Arbitrary Road Boundary Constraints," <i>arXiv preprint arXiv:2505.02395</i>, 2025.

<a href="https://arxiv.org/abs/2505.02395" target="_blank"><img src="https://img.shields.io/badge/-Preprint-b31b1b?logo=arXiv"></a> [![Jump to Fig. 5](https://img.shields.io/badge/Jump%20to-Fig.%205-blue)](#fig-safety-filter) <a href="https://github.com/bassamlab/SigmaRL/tree/1.4.0" target="_blank"><img src="https://img.shields.io/badge/-GitHub-181717?logo=GitHub"></a>
</div>

- **BibTeX**
  ```bibtex
  @article{xu2025realtime,
    title = {A Real-Time Control Barrier Function-Based Safety Filter for Motion Planning with Arbitrary Road Boundary Constraints},
    author = {Xu, Jianye and Che, Chang and Alrifaee, Bassam},
    journal = {arXiv preprint arXiv:2505.02395},
    year = {2025},
  }
  ```

- **Reproduce Experimental Results in the Paper:**

  - Git checkout to the corresponding tag using `git checkout 1.4.0`
  - Go to [this page](https://github.com/bassamlab/assets/blob/main/sigmarl/checkpoints/itsc25.zip) and download the zip file `itsc25.zip`. Unzip it, copy and paste the whole folder to the `checkpoints` folder at the **root** of this repository. The structure should be like this: `root/checkpoints/itsc25/`.
  - Run `sigmarl/evaluation_itsc25.py`.


## TODOs
- Improve safety
  - [ ] Integrating Control Barrier Functions (CBFs)
    - [x] Proof of concept with two agents (see the MTV-Based CBF paper [here](#3-mtv-based-cbf))
    - [x] High-Order CBFs (see the TTCBF paper [here](#4-truncated-taylor-cbf))
    - [x] Collision aovidance with road boundaries (see the CBF-Based Safety Filter paper [here](#5-cbf-based-safety-filter))
  - [ ] Integrating Model Predictive Control (MPC)
- Address non-stationarity
  - [x] Integrating prioritization (see the XP-MARL paper [here](#2-xp-marl))
- Effective observation design
  - [ ] Image-based representation of observations
  - [ ] Historic observations
  - [ ] Attention mechanism
- Misc
  - [x] OpenStreetMap support (see guidance [here](#customize-your-own-maps))
  - [x] Contribute our [CPM scenario](#fig-scenario-cpm) as an MARL benchmark scenario in VMAS (see news <a href="https://github.com/proroklab/VectorizedMultiAgentSimulator/releases/tag/1.4.2" target="_blank">here</a>)
  - [x] Update to the latest versions of Torch, TorchRL, and VMAS
  - [x] Support Python 3.11+

## Acknowledgments
This research was supported by the Bundesministerium für Digitales und Verkehr (German Federal Ministry for Digital and Transport) within the project "Harmonizing Mobility" (grant number 19FS2035A).
