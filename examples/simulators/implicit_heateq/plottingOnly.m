% Assumes that time step k is stored in U(:,k)

G = cartGrid([19, 24, 20]);   

% Assumes 150 time steps with interesting features.

dt = 4;

FigHandle = figure('Position', [600, 150, 800, 500]); % (x,y)

%subplot(2,2,1);
timestep = 1;
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
        timestep=4;
    elseif( i==2)
        timestep=9;
    elseif( i==3)
        timestep=17;
    end
    colorbar
end
%colorbar