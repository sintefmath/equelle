% Post-processing the 3D heat equation, using MRST for plotting the result
% For MRST, see: http://www.sintef.no/Projectweb/MRST/
%
% Argument:
%     folder (optional): folder containing Equelle output files.
%     default is this current folder.
%
% Usage with build as output-folder
% For octave from terminal: $ octave postProc.m build
% For octave from octave: :> outputDir = 'build'; postProc
% For matlab from matlab: :> outputDir = 'build'; postProc


cd <insert_local_path>/matlab/mrst-2013b/
startup
cd <insert_equelle_local_folder>/examples/simulators/heateq


isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;

if (isOctave)
    arg_list = argv();
    if (size(arg_list,1) > 0)
        disp('Changing directory')
        outputDir = arg_list{1};
    end
end

if (exist('outputDir', 'var')) 
    cd(outputDir);
end

% Create the grid:
G = cartGrid([19, 24, 20]);   
%G = cartGrid([8000, 1, 1]);
var = 'expU-';
ext = '.output';

files = dir(sprintf('%s*%s', var, ext));
T = size(files, 1);

% Map U on cartGrid
U = zeros(prod(G.cartDims), T);

for i = 0:1:T-1
    if (i < 10)
        file = sprintf('%s0000%d%s', var, i, ext);
    elseif (i < 100)
        file = sprintf('%s000%d%s', var, i, ext);
    elseif (i < 1000)
        file = sprintf('%s00%d%s', var, i, ext);
    end
    loading = i
    U(:,i+1) = load(file);
end

for i = 1:1:T
    plotCellData(G,  U(:,i));
    view(30,50);
    pause(0.1);
    plotting = i
end