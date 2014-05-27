disp('Start postprocess')

% find files
files = dir('u-*.output');
T = size(files,1)

u0 = load('u-00000.output');
N = size(u0, 1)

% Allocate memory for solutions
U = zeros(N,T);


for i = 0:T-1
    if (i < 10)
        loadFile = sprintf('u-0000%d.output', i);
    else
    	loadFile = sprintf('u-000%d.output', i); 
    end
    u = load(loadFile);
    U(:,i+1) = u;
end

x = 1:N;
x = x - 0.5;
for i = 1:T
    plot(x,U(:,i))
    axis( [0 N 0 1000] )
    xlabel('x')
    ylabel('temperature')

    pause(1.0)
end

disp('End post processing')