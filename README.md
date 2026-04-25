# LeRobot for Research (with RECAP)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange)](https://huggingface.co/)

Welcome to the LeRobot research repository, showcasing integration with the generalized RECAP methodology.

## Table of Contents
- [What is implemented here](#what-is-implemented-here)
- [Video Introduction](#video-introduction)
- [Quick Start](#quick-start)
  - [Installation & Prerequisites](#installation--prerequisites)
  - [Basic Offline Training](#basic-offline-training)
  - [Basic Inference](#basic-inference)
- [Advanced Documentation](#advanced-documentation)

## What is implemented here

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.

- **Feature 1:** Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia.
- **Feature 2:** Deserunt mollit anim id est laborum sed ut perspiciatis unde omnis.
- **Feature 3:** Iste natus error sit voluptatem accusantium doloremque laudantium.

## Video Introduction

This section typically contains a demonstration of the policy evaluating in the real world. 

<video src="https://github.com/placeholder/video.mp4" controls width="100%"></video>

*Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione.*

## Quick Start

### Installation & Prerequisites

To begin your journey with RECAP, you must first prepare the environment. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit.

```bash
git clone https://github.com/example/repo.git
cd repo
pip install -e .
```

### Basic Offline Training

Initial offline training is crucial for establishing baseline behavior before online fine-tuning. Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam.

```bash
python scripts/train_offline.py --config configs/base.yaml
```

### Basic Inference

To run the model on your hardware, execute the inference script. Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur.

```bash
python scripts/inference.py --checkpoint checkpoints/latest.pt
```

## Advanced Documentation
For details on how to use all the scripts (annotation, online training, advanced validation), please refer to the [detailed script usage documentation](docs/pi05_docs/usage.md).

For mathematics, logic, and deep dives into the codebase architecture (RECAP, action encodings, RTC, and buffer logic), please refer to the [architecture documentation](docs/pi05_docs/architecture.md).
