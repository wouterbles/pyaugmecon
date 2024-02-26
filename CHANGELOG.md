# Changelog

## 1.0.8

- Add method to check if user-provided model component names mask pyaugmecon-added component names
- Update LICENSE

## 1.0.7

- Update readme to include other solvers
- Fix Pandas warning

## 1.0.6

- Fix dependencies

## 1.0.5

- Update dependencies

## 1.0.4

- Correctly enqueue items such that the work can be redistributed when a process finishes its work

## 1.0.3

- Use new Pyomo 6.6.1 method to `close()` Gurobi and prevent duplicate instances, thanks @torressa

## 1.0.2

- Update dependencies

## 1.0.1

- Make QueueHandler work on MacOS #5, thanks @yvanoers

## 1.0.0

- Add new methods for solution retrieval

## 0.2.1

- Add Python 3.10 to supported versions

## 0.2.0

- Decision variables are now stored in a dictionary together with the objective values
- More details in the [documentation](https://github.com/wouterbles/pyaugmecon#pyaugmecon-solutions-details)

## 0.1.9

- Again fix package dependencies to prevent breaking changes
- Update dependencies

## 0.1.8

- Relax dependency requirements

## 0.1.7

- Remove default solver options when setting them to `None`

## 0.1.6

- Fix issue with mixed min/max objectives

## 0.1.5

- Fix dependency versions after issue with change in Pymoo API

## 0.1.4

- Incorrectly bumped version, no change from 0.1.3

## 0.1.3

- Trigger new release for Zenodo

## 0.1.2

- Move process timeout check to seperate thread to prevent a deadlock

## 0.1.1

- Add more detailed installation instructions to README and fix typos

- Add CHANGELOG

## 0.1.0

- 🎉 First published version!

- Alpha quality
