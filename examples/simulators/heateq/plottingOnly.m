% Utility script for plotting chosen timesteps in one figure.
% 
% Grid dimensions and chosen timesteps should be hard-coded here.
% 
% Requres that time step k is stored in U(:,k) (produced by bigPostProc.m),
% and also MRST (http://www.sintef.no/Projectweb/MRST/).

G = cartGrid([19, 24, 20]);   

% Assumes 150 time steps with interesting features.

dt = 0.5;

FigHandle = figure('Position', [600, 150, 800, 500]); % (x,y)

%subplot(2,2,1);
timestep = 4;
for i=1:4
    subplot(2,2,i)
    plotCellData(G, U(:,timestep))
    view(20,30);    
    titleText = sprintf('T = %2.1f sec, iter %d', timestep*dt, timestep);
    title(titleText)
    xlabel('x');
    ylabel('y');
    zlabel('z');
    if( i== 1) 
        timestep=30;
    elseif( i==2)
        timestep=70;
    elseif( i==3)
        timestep=140;
    end
    colorbar
end
%colorbar
