clf
[V,F] = readOBJ('../example_meshes/Bird.obj');
[Vch,Fch] = readOBJ('../example_meshes/Bird_ch.obj');
%V = -V;
com = centroid(V,F);
com_ch = centroid(Vch,Fch);
N = per_vertex_normals(V, F, 'Weighting','area');
N = N./normrow(N);
vertex_to_com = com - V;
vertex_to_com = vertex_to_com./normrow(vertex_to_com);
dp = dot(vertex_to_com,N,2);

shadingParams = {'FaceLighting','gouraud', 'FaceColor','interp'};
tsurf(F,V,"Cdata",ones(size(V,1),1),shadingParams{:}, ...
    'DiffuseStrength',0.5, 'SpecularStrength',0.2, 'AmbientStrength',0.3)
axis equal; axis off;
lights = camlight;
shading interp
% colorbar
light('Position',[0 0 -2],'Style','local');
hold on
plot3(com(1),com(2),com(3),".r",'MarkerSize',35)
% plot3(com_ch(1),com_ch(2),com_ch(3),".b",'MarkerSize',15)
v = V(807,:);
vec = (com-v)/norm(com-v)*20;
qvr(v,vec,'linewidth',6,"color","k","ShowArrowHead",1,"MaxHeadSize",3)
