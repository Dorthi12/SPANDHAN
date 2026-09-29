% MAIN - SPANDHAN Main Execution Pipeline
% Initializes configuration, loads data, executes DSP analysis, and calls ML classification.

clear; clc; close all;

% Add matlab directory and all subdirectories to MATLAB path
matlabRoot = fileparts(mfilename('fullpath'));
addpath(genpath(matlabRoot));

fprintf('=== SPANDHAN Pipeline Initializing ===\n');

% Load Configuration
cfg = config();

fprintf('Pipeline ready. Select DSP and ML modules to execute.\n');
