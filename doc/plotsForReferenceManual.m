%% This plotting script needs MRST to run

% Set up grid
G = cartGrid([3,3]);
G = removeCells(G, 2);
G = computeGeometry(G);

% Create some colors
ly = [1 1 0.5];
lb = [0.5 0.5 1];


%% Cells and faces enumerated
figure(1)
% Plot the grid
plotGrid(G,'FaceColor', ly, 'LineWidth', 1); axis off;
hold on;
% To improve readability, we extract the cell and face centroids as well as
% the coordinates of each node. These will be used for plotting the cells,
% faces and node indices respectively.
c_cent = G.cells.centroids;
f_cent = G.faces.centroids;
coords = G.nodes.coords;
% Add circles around the centroids of each cell
plot(c_cent(:,1), c_cent(:,2),'or','MarkerSize',30);
% Plot triangles around face centroids
%plot(f_cent(:,1), f_cent(:,2),'sg','MarkerSize',24);
% Plot squares around nodes
%plot(coords(:,1), coords(:,2),'db','MarkerSize',24);
%legend({'Grid', 'Cell', 'Face', 'Node'}, 'Location', 'SouthOutside', 'Orientation', 'horizontal')
alphabet = 'abcdefghijklmnopqrstuvwxyz';
% Plot cell/face centroids and nodes
text(c_cent(:,1)-0.04, c_cent(:,2), num2str((1:G.cells.num)'),'FontSize',24);
text(f_cent(:,1)+0.02, f_cent(:,2)+0.07, alphabet(1:G.faces.num)','FontSize',18);
%text(coords(:,1)-0.075, coords(:,2), num2str((1:G.nodes.num)'),'FontSize', 18);
%title('Grid structure')
hold off;

%%
export_fig grid-enumeration -pdf -transparent
