# Sudoku Solver using OpenCV

A Python script that extracts a Sudoku puzzle from a user-provided image and solves it using OpenCV.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)  

## Introduction

This project implements a Sudoku solver using OpenCV. It allows users to input an image containing a Sudoku puzzle, extracts the puzzle, and solves it.

## Features

- Image processing with OpenCV for puzzle extraction.
- Sudoku solving algorithm.
- User-friendly command-line interface.

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python (version 3.6 or higher)
- OpenCV
- NumPy

```bash
pip install opencv-python numpy streamlit
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/sudoku-solver.git
```

Navigate to the project directory:

```bash
cd sudoku-solver
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app with the following command:

```bash
streamlit run sudoku_solver_app.py
```

Visit the provided URL in your web browser to interact with the Sudoku solver.
