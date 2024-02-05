# Rignet result notes
* Mesh 304 looks good but the placement of the com is totally off.
* Tried another mesh and the dop products look totally off, I don't understand why. Then, I changed every objects origin to (0,0,0) and the dot products looked more reasonable.
* Just after loading the kangaroo I get:<br/>
NEW COMPARISON<br/>
COM rel error %:10000.49766465151<br/>
COM: [2.02714782e-05 2.85666436e-01 2.56143689e-01]<br/>
COM_tri: [ 2.02737687e-07 -2.56143733e-03  2.85666516e-03]<br/>
<br/>
COM_CH rel error%:10031.228672815603<br/>
COM_CH_tri: [-7.38704369e-08 -1.82971411e-03  2.95055493e-03]<br/>
COM_CH: [2.17184424e-06 2.95661181e-01 1.84029162e-01]<br/>
<br/>
tri_vn:[-0.35764557 -0.84314294  0.40149673]<br/>
vn:<Vector (-0.3576, -0.8431, 0.4015)><br/>
Rel error vn %7.427013015181565e-05<br/>
<br/>
Point: <Vector (-0.0004, -0.0040, 0.0044)><br/>
Point_tri: [-0.00044383 -0.00396416  0.00435139]<br/>
Rel error point%0.0<br/>
* After undoing the 0.01 transformations I get: <br/>
COM rel error %:141.42133452697198<br/>
COM: [2.02714782e-05 2.85666436e-01 2.56143689e-01]<br/>
COM_tri: [ 2.02738666e-05 -2.56143733e-01  2.85666521e-01]<br/>
<br/>
COM_CH rel error%:141.52205926386142<br/>
COM_CH_tri: [-7.38658828e-06 -1.82971423e-01  2.95055491e-01]<br/>
COM_CH: [2.17184424e-06 2.95661181e-01 1.84029162e-01]<br/>
<br/>
tri_vn:[-0.35764605 -0.84314225  0.40149777]<br/>
vn:<Vector (-0.3576, -0.8431, 0.4015)><br/>
Rel error vn %0.00020027082452545687<br/>
<br/>
Point: <Vector (-0.0444, -0.3964, 0.4351)><br/>
Point_tri: [-0.044383   -0.39641601  0.435139  ]<br/>
Rel error point%0.0<br/>
<br/>
p2com_tri: [ 0.2117084   0.66879796 -0.7126632 ]<br/>
dp_tri: -0.9257411712817658<br/>
p2com:[ 0.06284302  0.965337   -0.25332832]<br/>
dp:-0.9381033698579346<br/>
* I added a step in the computation of com, com_ch where I multiply by matrix_world and now the errors are small! I also tested it