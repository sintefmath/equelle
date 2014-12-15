% Script to make a bottom elevation 
% 
% Grid consists of 100 x 100 cells, and thus 101 x 101 faces
% We span the bottom topography across an array of 201 x 201 entries

B = zeros(201,201);

% x is array with points in space.
x = 0:0.5:100;
y = 0:0.5:100;

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

for i = 1:201
    for j = 1:201
        B(i,j) = 1/300*i + 1/200*j + 0.3*sin((1/170)*i*2*pi) + 0.2*sin((1/170)*j*2*pi);
        r = sqrt((i-100)^2 + (j-100)^2);
        if ( r < 85 && r > 65) 
            B(i,j) = B(i,j) + 0.3*sin( 0.05*(r-65)*pi);
        end
    end
end

surf(x, y, B)
axis([0 100 0 100 0 3])

% store B at cells
(2:2:10)
bmid = B(2:2:200, 2:2:200);
bmidsize = size(bmid)
save('bshore_mid.mat', 'bmid', '-ascii')

% store B at south face
(1:2:10)
bsouth = B(1:2:199, 2:2:200);
bsouthSize = size(bsouth)
save('bshore_south.mat', 'bsouth', '-ascii')

% store B at north face
(3:2:10)
bnorth = B(3:2:201, 2:2:200);
bnorthsize = size(bnorth);
save('bshore_north.mat', 'bnorth', '-ascii')

bwest = B(2:2:200, 1:2:199);
bwestSize = size(bwest)
save('bshore_west.mat', 'bwest', '-ascii')

beast = B(2:2:200, 3:2:201);
beastSize = size(beast)
save('bshore_east.mat', 'beast', '-ascii')

% Create flat surface
% Flat surface -> h + b = const
%H = 1.5*ones(100,100);
%for i=-20:20
%    for j=-20:20
%        if ( i*i + j*j < 15*15 )
%            H(i+50,j+50) = 2.5;
%        end
%    end
%end

% Quarter filled domain
%quarter = zeros(50,50);
H = zeros(100,100);
for i=1:50
    for j=1:50
        H(101-i,101-j) = 3*exp(-(((0.8*i+0.9*j))/50)^2);
    end
end
%H = zeros(100,100);
%H(1:50, 1:50) = quarter(1:50, 1:50);

% Remove negative water depths
H = H - bmid;
for i=1:100
    for j=1:100
        if (H(i,j) < 0 )
            H(i,j) = 0;
        end
    end
end
       
hold on
surf(1:100, 1:100, H + bmid);
hold off
%save('shore_surf.mat', 'H', '-ascii')
save('quarter_surf.mat', 'H', '-ascii')

% Init velocities
initU = zeros(100, 100);
save('initU_a.mat', 'initU', '-ascii')


% Write timesteps ( prosjektet needs 2500 steps of 0.03
timesteps = 1000
dt = 0.03
%dt = 0.2

t = ones(1,timesteps).*dt;
save('timesteps.mat', 't', '-ascii')
