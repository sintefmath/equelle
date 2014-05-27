


lowering = 0.00

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


timestep = 50;
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


timestep = 120;
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



timestep = 200;
subplot(3,2,4)
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

timestep = 450;
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


timestep = 800;
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