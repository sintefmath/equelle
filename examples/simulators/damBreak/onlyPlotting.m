

animation=0;
if (animation == 1)
    for i = 1:su:T
    %    for j=1:100
    %        matH(:,j) = H((j-1)*100+1:j*100,i)-0.06;
    %    end
    %subplot(2,1,1)
    %surf(x(10:90), y(10:90), matH(10:90, 10:90));
    %h_surf = surf(x, y, matH, 'faceColor', [0 0 1],'edgeColor', 'none', 'faceLighting', 'gouraud');
    %colormap('winter')
    %axis([0 100 0 100 0 max(q10)+1])
    %alpha(h_surf, 0.6);
    %hold on
    %light('Position',[1 0.7 0.5],'Style','infinite');
    %surf(x, y, matB, 'faceColor', [1 1 0], 'edgeColor', 'none', 'faceLighting', 'gouraud');
    %%colormap('copper')
    %hold off
    %colormap('')
    %view(160, 60);
    %time(i)
    %i
    %subplot(2,1,1)
    plot(x, B, 'k--', x, H(:, i), 'b')
    axis( [ 0 200 0 5])
    xlabel('Distance, x')
    ylabel('Surface elevation, \omega')
    legend('Bottom', 'Water surface')
    %subplot(2,1,2)
    %plot(x, U)
    %axis( [0 200 min(min(U))-1 max(max(U))+1 ] ) 
    if (mod(i, 50)==0)
        text = sprintf('time : %f seconds', i*0.03)
    end
    pause(0.03)
    
    %axis([0 100 0 10])
    %subplot(2,1,2)
    %plot(x,U(:,i))
    %axis([0 100 -7 7])
    %pause(0.1)
    
    end % end for loop

    
end % end animation

dt = 0.03;

figure(1)
subplot(6,1,1)
plot(x, B, 'k--', x, H(:, 1), 'b')
axis( [ 0 200 0 5])
%xlabel('Distance, x')
ylabel('\omega')
legend('Bottom', 'Water surface')
title('T = 0.0')

timestep = 200;
time = (timestep*dt);
subplot(6,1,2)
plot(x, B, 'k--', x, H(:,timestep), 'b')
axis( [ 0 200 0 5])
%xlabel('Distance, x')
ylabel('\omega')
%legend('Bottom', 'Water surface')
titleText = sprintf('T = %3.1f s', time);
title(titleText)

timestep = 700;
time = (timestep*dt);
subplot(6,1,3)
plot(x, B, 'k--', x, H(:,timestep), 'b')
axis( [ 0 200 0 5])
%xlabel('Distance, x')
ylabel('\omega')
%legend('Bottom', 'Water surface')
titleText = sprintf('T = %3.1f s', time);
title(titleText)

timestep = 1500;
time = (timestep*dt);
subplot(6,1,4)
plot(x, B, 'k--', x, H(:,timestep), 'b')
axis( [ 0 200 0 5])
%xlabel('Distance, x')
ylabel('\omega')
%legend('Bottom', 'Water surface')
titleText = sprintf('T = %3.1f s', time);
title(titleText)


timestep = 3000;
time = (timestep*dt);
subplot(6,1,5)
plot(x, B, 'k--', x, H(:,timestep), 'b')
axis( [ 0 200 0 5])
%xlabel('Distance, x')
ylabel('\omega')
%legend('Bottom', 'Water surface')
titleText = sprintf('T = %3.1f s', time);
title(titleText)


timestep = 5700;
time = (timestep*dt);
subplot(6,1,6)
plot(x, B, 'k--', x, H(:,timestep), 'b')
axis( [ 0 200 0 5])
%xlabel('Distance, x')
ylabel('\omega')
%legend('Bottom', 'Water surface')
titleText = sprintf('T = %3.1f s', time);
title(titleText)



%% Plot velocities:
figure(2)
%subplot(6,1,1)
%plot(x, U(:, 1), 'b')
%axis( [ 0 200 -3 10])
%grid on
%%xlabel('Distance, x')
%ylabel('u')
%legend('water velocity')
%title('T = 0.0')

%timestep = 200;
%time = (timestep*dt);
%subplot(6,1,2)
%plot(x, U(:,timestep), 'b')
%grid on
%axis( [ 0 200 -3 10])
%%xlabel('Distance, x')
%ylabel('u')
%%legend('Bottom', 'Water surface')
%titleText = sprintf('T = %3.1f s', time);
%title(titleText)

timestep = 700;
time = (timestep*dt);
subplot(3,1,1)
plot(x, U(:,timestep), 'b')
grid on
axis( [ 0 200 -3 10])
%xlabel('Distance, x')
ylabel('u')
%legend('Bottom', 'Water surface')
titleText = sprintf('Water velocity at T = %3.1f s', time);
title(titleText)

timestep = 1500;
time = (timestep*dt);
subplot(3,1,2)
plot(x, U(:,timestep), 'b')
grid on
axis( [ 0 200 -3 10])
%xlabel('Distance, x')
ylabel('u')
%legend('Bottom', 'Water surface')
titleText = sprintf('Water velocity at T = %3.1f s', time);
title(titleText)


timestep = 3000;
time = (timestep*dt);
subplot(3,1,3)
plot(x, U(:,timestep), 'b')
grid on
axis( [ 0 200 -3 3])
%xlabel('Distance, x')
ylabel('u')
%legend('Bottom', 'Water surface')
titleText = sprintf('Water velocity at T = %3.1f s', time);
title(titleText)


%timestep = 5700;
%time = (timestep*dt);
%subplot(6,1,6)
%plot( x, U(:,timestep), 'b')
%grid on
%axis( [ 0 200 -5 5])
%%xlabel('Distance, x')
%ylabel('u')
%%legend('Bottom', 'Water surface')
%titleText = sprintf('T = %3.1f s', time);
%title(titleText)





