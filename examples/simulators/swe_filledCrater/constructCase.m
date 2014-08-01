% Script to make a bottom elevation 
% 
% Grid consists of 100 x 100 cells, and thus 101 x 101 faces
% We span the bottom topography across an array of 201 x 201 entries

format compact;

nx = 231;
ny = 198;

B = zeros(ny*2 +1,nx*2 + 1);

% x is array with points in space.
x = 0:0.5:nx;
y = 0:0.5:ny;

% Add stuff to B:
%smallX = [-4:0.2:4];
%smallY = [-4:0.2:4];
%bump = zeros(41, 41);
%for i = 1:41
%    for j = 1:41
%        bump(i,j) = 1.3*exp(- ( (smallX(i)*2)^2 + (smallY(j)/1.3)^2)/4);
%    end
%end

%sizeBump = size(bump)
%size(B(2:2:200, 2:2:200))
%B(85:85+sizeBump(1)-1, 10:10+sizeBump(2)-1) = bump;
%B(70:70+sizeBump(1)-1, 80:80+sizeBump(2)-1) = bump;
%B(20:20+sizeBump(1)-1, 100:100+sizeBump(2)-1) = bump';

%for i = 1:201
%    for j = 1:201
%        B(i,j) = 1/350*i + 1/300*j + 0.2*sin((1/201)*i*2*pi) + 0.1*sin((1/201)*j*2*pi);
%        r = sqrt((i-100)^2 + (j-100)^2);
%        if ( r < 85 && r > 65) 
%            B(i,j) = B(i,j) + 0.5*sin( 0.05*(r-65)*pi);
%        end
%    end
%end

% Bottom will be of the form of a dome
% This means a sinus with zeros at all boundaries
for i=1:2*nx + 1
    for j=1:2*ny + 1
        %B(j,i) = (i-1) + (j-1)*(2*nx+1);
        B(j,i) = sin(x(i)*pi/nx)*sin(y(j)*pi/ny);
        % B(y,x)
    end
end
surf(x, y, B)
shading('interp')
axis([0 nx 0 ny 0 3])

% store B at cells
(2:2:10)
bmid = B(2:2:2*ny, 2:2:2*nx);
bmidsize = size(bmid)
save('bCase_mid.mat', 'bmid', '-ascii')


% store B at south face
(1:2:10)
bsouth = B(1:2:2*ny-1, 2:2:2*nx);
bsouthSize = size(bsouth)
save('bCase_south.mat', 'bsouth', '-ascii')

% store B at north face
(3:2:10)
bnorth = B(3:2:2*ny+1, 2:2:2*nx);
bnorthsize = size(bnorth);
save('bCase_north.mat', 'bnorth', '-ascii')

bwest = B(2:2:2*ny, 1:2:2*nx-1);
bwestSize = size(bwest)
bwest = bwest';
save('bCase_west.mat', 'bwest', '-ascii')

beast = B(2:2:2*ny, 3:2:2*nx+1);
beastSize = size(beast)
save('bCase_east.mat', 'beast', '-ascii')

% Create flat surface
% Flat surface -> h + b = const

H = 2.5*ones(ny,nx); % -bmid;
H = H - bmid;
hold on

surf(1:nx, 1:ny, H + bmid);
shading('interp')
hold off
H = H';
save('case_surf.mat', 'H', '-ascii')


% Init velocities
%initU = zeros(nx, ny);
%save('initUCase_a.mat', 'initU', '-ascii')

% Write timesteps ( prosjektet needs 2500 steps of 0.03
timesteps = 10
dt = 0.03
%dt = 0.2

t = ones(1,timesteps).*dt;
save('timestepsCase.mat', 't', '-ascii')
