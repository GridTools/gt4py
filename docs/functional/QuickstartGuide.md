---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# GT4Py User Quickstart Guide

+++

## Info for writing the guide
Goals:
- Read by someone completely new
- Focus on unstructured
They will be able to:
- Install GT4Py
- Write simple code
- Understand basic concepts of GT4Py
Examples could be:
- Laplacian
    - Main concepts: fields, field operators, programs, reductions
- Realistic example: diffusion, advection, solve some simple PDE on a 2D grid

+++

## Installation

GT4Py is distributed as a Python package and can be installed directly from GitHub:

```{code-cell} ipython3
#! pip install git+https://github.com/gridtools/gt4py.git@functional
```

## Introduction

What does GT4Py do?
- Defines some Python DSL
  - Which is a subset of python
  - With some high level concepts for weather modeling
- Runs DSL directly in Python
- Transpiles DSL to C++ or CUDA
- Optimizes transpiled codeso that it's super fast

What is GT4Py for?
- Implementing PDEs on structured and unstructured grids
- Specialized for numerical weather modeling

+++

## Concepts

### Basics

Example:
Fields, operators and programs:

```{code-cell} ipython3
@field_operator
def add(field1, field2):
    pass

@program
def compute_sum():
    pass

add()
```

Explanation:

#### Fields

#### Field operators

#### Programs

#### Interfacing with enclosing Python code

+++

### More advanced

E2V, V2E, reductions

+++

## Examples

+++

### Space derivatives

```{code-cell} ipython3
@field_operator
def ddx():
    pass

@program
def compute_ddx():
    pass

get_ddx()
```

### Diffusion

```{code-cell} ipython3
@field_operator
def op1():
    pass

@field_operator
def op1():
    pass

@program
def diffuse():
    pass

timestep = 0.01
endtime = 5
for time in range(0:endtime:timestep):
    diffuse()
```
