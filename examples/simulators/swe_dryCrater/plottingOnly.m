% Utility script for plotting chosen timesteps in one figure.
% 
% Grid dimensions and chosen timesteps should be hard-coded here.
% 
% Requres that regular post processing already is done.


lowering = 0.04

animation = 0;
if (animation == 1)
for i = 1:T
    for j=1:100
        matH(:,j) = H((j-1)*100+1:j*100,i)-lowering;
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
    view(280,40);
    %time(i)
    i
    pause(0.01)
    
    %axis([0 100 0 10])
    %subplot(2,1,2)
    %plot(x,U(:,i))
    %axis([0 100 -7 7])
    %pause(0.1)
end

end % animation

dt=0.03;

figure(2)
timestep = 1;
subplot(3,2,1)
for j=1:100
    matH(:,j) = H((j-1)*100+1:j*100,timestep)-lowering;
end
h_surf = surf(x, y, matH, 'faceColor', [0 0 1],'edgeColor', 'none', 'faceLighting', 'gouraud');
axis([0 100 0 100 0 max(q10)+1])
hold on
light('Position',[0.3 0.7 0.5],'Style','infinite');
surf(x, y, matB, 'faceColor', [1 1 0], 'edgeColor', 'none', 'faceLighting', 'gouraud');
hold off
titleText = sprintf('T = %3.2f', 0.0);
title(titleText)
zlabel('\omega')
view(280,40);


timestep = 167;
subplot(3,2,2)
for j=1:100
    matH(:,j) = H((j-1)*100+1:j*100,timestep)-lowering;
end
h_surf = surf(x, y, matH, 'faceColor', [0 0 1],'edgeColor', 'none', 'faceLighting', 'gouraud');
axis([0 100 0 100 0 max(q10)+1])
hold on
light('Position',[1 0.7 0.5],'Style','infinite');
surf(x, y, matB, 'faceColor', [1 1 0], 'edgeColor', 'none', 'faceLighting', 'gouraud')
hold off
titleText = sprintf('T = %3.2f', timestep*dt);
title(titleText)
zlabel('\omega')
view(280,40);


timestep = 500;
subplot(3,2,3)
for j=1:100
    matH(:,j) = H((j-1)*100+1:j*100,timestep)-lowering;
end
h_surf = surf(x, y, matH, 'faceColor', [0 0 1],'edgeColor', 'none', 'faceLighting', 'gouraud');
axis([0 100 0 100 0 max(q10)+1])
hold on
light('Position',[1 0.7 0.5],'Style','infinite');
surf(x, y, matB, 'faceColor', [1 1 0], 'edgeColor', 'none', 'faceLighting', 'gouraud');
hold off
titleText = sprintf('T = %3.2f', timestep*dt);
title(titleText)
zlabel('\omega')
view(280,40);



timestep = 1000;
subplot(3,2,4)
for j=1:100
    %matH(:,j) = H((j-1)*100+1:j*100,timestep)-lowering;
    matH(:,j) = H((j-1)*100+1:j*100,timestep);
    for i=1:100
        if ( abs(matH(i,j) - matB(i,j)) < 0.05 )
            matH(i,j) = matB(i,j) - 0.01;
        end
    end
end
h_surf = surf(x, y, matH, 'faceColor', [0 0 1],'edgeColor', 'none', 'faceLighting', 'gouraud');
axis([0 100 0 100 0 max(q10)+1])
hold on
light('Position',[1 0.7 0.5],'Style','infinite');
surf(x, y, matB, 'faceColor', [1 1 0], 'edgeColor', 'none', 'faceLighting', 'gouraud');
hold off
titleText = sprintf('T test = %3.2f', timestep*dt);
title(titleText)
zlabel('\omega')
view(280,40);

timestep = 1500;
subplot(3,2,5)
for j=1:100 
    matH(:,j) = H((j-1)*100+1:j*100,timestep)-lowering;
end
h_surf = surf(x, y, matH, 'faceColor', [0 0 1],'edgeColor', 'none', 'faceLighting', 'gouraud');
axis([0 100 0 100 0 max(q10)+1])
hold on
light('Position',[1 0.7 0.5],'Style','infinite');
surf(x, y, matB, 'faceColor', [1 1 0], 'edgeColor', 'none', 'faceLighting', 'gouraud');
hold off
titleText = sprintf('T = %3.2f', timestep*dt);
title(titleText)
zlabel('\omega')
view(280,40);


timestep = 2500;
subplot(3,2,6)
for j=1:100
    matH(:,j) = H((j-1)*100+1:j*100,timestep)-lowering;
end
h_surf = surf(x, y, matH, 'faceColor', [0 0 1],'edgeColor', 'none', 'faceLighting', 'gouraud');
axis([0 100 0 100 0 max(q10)+1])
hold on
light('Position',[1 0.7 0.5],'Style','infinite');
surf(x, y, matB, 'faceColor', [1 1 0], 'edgeColor', 'none', 'faceLighting', 'gouraud');
hold off
titleText = sprintf('T = %3.2f', timestep*dt);
title(titleText)
zlabel('\omega')
view(280,40);




% Seperate figure

figure(3)
timestep = 1;
for j=1:100
    matH(:,j) = H((j-1)*100+1:j*100,timestep)-lowering;
end
h_surf = surf(x, y, matH, 'faceColor', [0 0 1],'edgeColor', 'none', 'faceLighting', 'gouraud');
axis([0 100 0 100 0 max(q10)+1])
hold on
light('Position',[0.3 0.7 0.5],'Style','infinite');
surf(x, y, matB, 'faceColor', [1 1 0], 'edgeColor', 'none', 'faceLighting', 'gouraud');
hold off
titleText = sprintf('T = %3.2f', 0.0);
title(titleText)
zlabel('\omega')
view(280,40);