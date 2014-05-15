%
% This is a script designed for writing Dirichlet boundary conditions 
% for the heat equation simulator written in Equelle.
% 
% This generator is based on a Cartesian 3D grid of size (nx,ny,nz) 
% and gives indices and values on the faces with normal in (pos and neg)
% x-direction. Other faces are faces with zero flux.
%

format compact;

nx = 100;
ny = 111;
nz = 50;
timesteps = 20
dt = 0.2

t = ones(1,timesteps).*dt;
save('timesteps.mat', 't', '-ascii')


indices = zeros(1,2*ny*nz);
ind = 2;
for i=1:ny*nz - 1
    indices(ind) = i*(nx + 1) -1;
    indices(ind+1) = i*(nx + 1);
    ind = ind + 2;
end
indices(2*ny*nz) = ny*nz*(nx + 1) -1;
indices = indices';
fid = fopen('dir_bnd.mat', 'w');
for i=1:size(indices)
    fprintf(fid, '%d\n', indices(i));
end
fclose(fid);


% VALUES:
% [ cold warm] -> [warm cold]
% Uniform in nz, {in/de}creasing in dy
% Max: 1000, Min: 0
val_y = linspace(1000, 0, ny);
dir_vals = zeros(nz, ny);
for y=1:ny
    dir_vals(:,y) = val_y(y);
end
%dir_vals
dirvalsList = zeros(1,2*ny*nz);
for y=1:ny
    for z=1:nz  
        % x = 0 correct way
        dirvalsList(2*((y-1)*nz + z) -1) = dir_vals(z,y);
        % x = 0 opposite way
        dirvalsList(2*((y-1)*nz + z)) = dir_vals(z, ny-y+1);
    end
end
%dirvalsList
dirvalsList = dirvalsList';
save('dir_val.mat', 'dirvalsList', '-ascii')

disp('Num cells: ')
nx*ny*nz

disp('Num faces:')
numFaces = (nx+1)*ny*nz + nx*(ny+1)*nz + nx*ny*(nz+1)

size_dirvals = size(dirvalsList)
size_indices = size(indices)