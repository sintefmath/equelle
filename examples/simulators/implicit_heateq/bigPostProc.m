format compact;

% Startup MRST:
cd /home/havahol/programs/matlab/mrst-2013b/
startup
cd /home/havahol/sintefEquelle/equelle/examples/simulators/implicit_heateq


% Create the grid:
G = cartGrid([19,24,20]);   
dt = 4;
%G = cartGrid([8000, 1, 1]);
var = 'u-';
ext = '.output';

files = dir(sprintf('%s*%s', var, ext));
T = size(files, 1);

% Map U on cartGrid
U = zeros(prod(G.cartDims), T);

for i = 0:T-1
    if (i < 10)
        file = sprintf('%s0000%d%s', var, i, ext);
    elseif (i < 100)
        file = sprintf('%s000%d%s', var, i, ext);
    end
    loading = i
    U(:,i+1) = load(file);
end

for (i = 1:T)
    titletext = sprintf('T = %2.1f', dt*i)
    plotCellData(G,  U(:,i));
    view(30,50);
    title(titletext)
    pause(0.5);
    plotting = i
end