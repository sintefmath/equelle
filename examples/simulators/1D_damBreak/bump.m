% Script to make a bottom elevation 
% 
% Grid consists of 1 x 200 cells, and thus 1 x 201 faces
% We span the bottom topography across an array of 2 x 401 entries


% timesteps:
timesteps = 1000;
t = ones(1,timesteps)*0.03;
save('timesteps.mat', 't', '-ascii')


B = 2*ones(401,3);

% x is array with points in space.
x = 0:0.5:200;
y = 0:0.5:1;

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

for i = 1:401
    for j=1:3
        if abs(i - 100) < 20
            B(i,j) = B(i,j) + 0.5*(cos(pi*(i-100)/20) + 1);
        end
        if ( i > 200 && i < 301) 
            B(i,j) = B(i,j) - 2*sin(pi*(i-201)/200);
        end
        if ( i > 300)
            B(i,j) = 0;
        end
    end
end


%B(i,j) = 1/300*i + 1/200*j + 0.3*sin((1/170)*i*2*pi) + 0.2*sin((1/170)*j*2*pi);
%        r = sqrt((i-100)^2 + (j-100)^2);
%        if ( r < 85 && r > 65) 
%           B(i,j) = B(i,j) + 0.3*sin( 0.05*(r-65)*pi);
%       end
%    end
%end

size(x)
size(y)
size(B)
surf(x, y, B')
axis([0 200 0 1 0 3])

% store B at cells
(2:2:10)
bmid = B(2:2:400, 2);
bmidsize = size(bmid)
save('bdam_mid.mat', 'bmid', '-ascii')

% store B at south face
(1:2:10)
bsouth = B(2:2:400, 1);
bsouthSize = size(bsouth)
save('bdam_south.mat', 'bsouth', '-ascii')

% store B at north face
(3:2:10)
bnorth = B(2:2:400, 3);
bnorthsize = size(bnorth);
save('bdam_north.mat', 'bnorth', '-ascii')

bwest = B(1:2:399, 2);
bwestSize = size(bwest)
save('bdam_west.mat', 'bwest', '-ascii')

beast = B(3:2:401, 2);
beastSize = size(beast)
save('bdam_east.mat', 'beast', '-ascii')

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
H = zeros(200,1);
for i=1:200
    if (i < 100)
        H(i) = 4;
    else
        H(i) = 0;
    end
end
%H = zeros(100,100);
%H(1:50, 1:50) = quarter(1:50, 1:50);

% Remove negative water depths
H = H - bmid;
for i=1:200
    if (H(i) < 0 )
        H(i) = 0;
    end
end
figure(1)
plot(1:200, bmid, 1:200, bwest, 1:200, beast)
axis([0 200 0 (max(H)+max(bmid))])
hold on
%surf(1:100, 1:100, H + bmid);
plot(1:200, H+bmid);
hold off
%save('shore_surf.mat', 'H', '-ascii')
save('dam_surf.mat', 'H', '-ascii')

U0 = zeros(1,200);
V0 = zeros(1,200);
save('zeroU.mat', 'U0', '-ascii')
save('zeroV.mat', 'V0', '-ascii')
