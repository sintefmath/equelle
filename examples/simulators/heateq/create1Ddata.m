%
%
% This script is written to create the datafiles for 1D Dirichlet
% conditions for the heat equation.
%

format compact
% Number of cells:
N = 8000;

% Number of timesteps:
T = 20;

% Boundary conditions:
dir_val = [0 1000]
save('dir_val1D.mat', 'dir_val', '-ascii')

% Boundary faces:
dir_bnd = [0 N]
fid = fopen('dir_bnd1D.mat', 'w');
fprintf(fid, '%d\n', dir_bnd(1));
fprintf(fid, '%d\n', dir_bnd(2));
fclose(fid);

% timesteps:
t = ones(1,T)*0.5;
save('timesteps1D.mat', 't', '-ascii')

num_faces = N*4 + (N+1)
