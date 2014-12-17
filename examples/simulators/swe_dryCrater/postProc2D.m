% Post-processing of 2D shallow water equation for hard-coded grid size.
%
% Argument:
%     folder (optional): folder containing Equelle output files.
%     default is this current folder.
%
% Usage with build as output-folder
% For octave from terminal: $ octave postProc.m build
% For octave from octave: :> outputDir = 'build'; postProc
% For matlab from matlab: :> outputDir = 'build'; postProc


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



% find number of output files = number of timesteps
files = dir('q1*.output');
T = size(files, 1);

% load first timestep to see how many cells we have
q10 = load('q1-00000.output');
%q20 = load('q2-00000.output');
N = size(q10, 1);
nx = 100;
ny = 100;

H = zeros(N, T);
B = load('bottom-00000.output');
b_north_vec = load('b_north-00000.output');
b_south_vec = load('b_south-00000.output');
b_east_vec = load('b_east-00000.output');
b_west_vec = load('b_west-00000.output');
%U = zeros(N, T);
%V = zeros(N, T);
time = zeros(N,1);

%figure()
for i = 1:T
    if (i-1 < 10)
        loadFile_q1 = sprintf('q1-0000%d.output', i-1);
        %loadFile_q2 = sprintf('q2-0000%d.output', i-1);
        %loadFile_q3 = sprintf('q3-0000%d.output', i-1);
    elseif (i-1 < 100)
        loadFile_q1 = sprintf('q1-000%d.output', i-1);
        %loadFile_q2 = sprintf('q2-000%d.output', i-1);
        %loadFile_q3 = sprintf('q3-000%d.output', i-1);
    elseif (i-1 < 1000)
        loadFile_q1 = sprintf('q1-00%d.output', i-1);
        %loadFile_q2 = sprintf('q2-00%d.output', i-1);
        %loadFile_q3 = sprintf('q3-00%d.output', i-1);
    else 
        loadFile_q1 = sprintf('q1-0%d.output', i-1);
        %loadFile_q2 = sprintf('q2-0%d.output', i-1);
        %loadFile_q3 = sprintf('q3-0%d.output', i-1);
    end
    h = load(loadFile_q1);
    %hu = load(loadFile_q2);
    %hv = load(loadFile_q3);
    %u = hu./h;
    %v = hv./h;
    H(:,i) = h;
    %U(:,i) = u;
    %V(:,i) = v;
    
    % Evaluate the timestep criteria
    %timeCrits = zeros(1,2);
    %timeCrits(1) = 1/max([max(abs(u + sqrt(9.81*h))), max(abs(u - sqrt(9.81*h)))]);
    %timeCrits(2) = 1/max([max(abs(v + sqrt(9.81*h))), max(abs(v - sqrt(9.81*h)))]);
    %time(i) = 0.25*min(timeCrits);
    
end
    
x = 0.5:1:nx-0.5;
y = 0.5:1:ny-0.5;

matH = zeros(100,100);
matB = zeros(100,100);
b_north = 0.3*ones(100,100);
b_south = 0.3*ones(100,100);
b_east = 0.3*ones(100,100);
b_west = 0.3*ones(100,100);
for j=1:100
    matB(:, j) = B((j-1)*100+1:j*100);
    b_north(:, j) = b_north_vec((j-1)*100+1:j*100);
    b_south(:, j) = b_south_vec((j-1)*100+1:j*100);
    b_east(:, j) = b_east_vec((j-1)*100+1:j*100);
    b_west(:, j) = b_west_vec((j-1)*100+1:j*100);
end
for i = 1:T
    for j=1:100
        matH(:,j) = H((j-1)*100+1:j*100,i)-0.027;
    end
    %subplot(2,1,1)
    %surf(x(10:90), y(10:90), matH(10:90, 10:90));
    h_surf = surf(x, y, matH, 'faceColor', [0 0 1],'edgeColor', 'none', 'faceLighting', 'gouraud');
    %colormap('winter')
    axis([0 100 0 100 0 max(q10)+1])
    %alpha(h_surf, 0.6);
    hold on
    light('Position',[1 0.7 0.5],'Style','infinite');
    surf(x, y, matB, 'faceColor', [1 1 0], 'edgeColor', 'none', 'faceLighting', 'gouraud');
    %colormap('copper')
    hold off
    %colormap('')
    view(30,40);
    %time(i)
    i
    pause(0.02)
    
    %axis([0 100 0 10])
    %subplot(2,1,2)
    %plot(x,U(:,i))
    %axis([0 100 -7 7])
    %pause(0.1)
end
