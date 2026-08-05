%% hexa_moment_set.m
%  Admissible (Mx, My) moment set of a co-planar hexarotor at FIXED total
%  thrust, as the commanded yaw moment Mz varies.
%
%  Allocator model : pseudo-inverse followed by per-rotor clamping
%       T_i   = Tbar + dT_i ,   dT = G*[Mx;My;Mz] ,   G = pinv(K)(:,2:4)
%       -dn  <= dT_i <= up     (per-rotor box constraint)
%  For a fixed Mz the feasible (Mx,My) region is the 2-D polytope
%       A*[Mx;My] <= b(Mz)
%  with A = [G(:,1:2); -G(:,1:2)] and b = [up-G(:,3)*Mz ; dn+G(:,3)*Mz].
%  The polygon is obtained by exact vertex enumeration (all constraint
%  pairs), so no gridding / sampling error is involved.
%
%  Outputs
%   Fig.1  overlaid (Mx,My) slices for several Mz
%   Fig.2  area and per-axis extent versus Mz
%   Fig.3  optional animation over Mz
%
%  kay01-kwon / drone_control_pkgs

clear; clc; close all;

%% ------------------------- Parameters -------------------------------
P.C_T      = 1.3175e-7;    % thrust coefficient        [N/rpm^2]
P.k_m      = 0.01569;      % rotor drag(moment) coeff  [m]
P.L        = 0.265;        % arm length                [m]
P.rpm_min  = 2000;         % rotor lower limit         [rpm]
P.rpm_max  = 7900;         % rotor upper limit         [rpm]
P.mass_kgf = 3.066;        % total thrust in kgf (T_tot = mass_kgf*g)
P.g        = 9.81;

ANIMATE = false;           % set true for Fig.3 animation

%% ------------------- Allocation matrix K ----------------------------
c = cos(pi/3);  s = sin(pi/3);
ly = P.L*[ c;  1;  c; -c; -1; -c];      % roll  arms
lx = P.L*[ s;  0; -s; -s;  0;  s];      % pitch arms
sg = [-1; 1; -1; 1; -1; 1];             % rotor spin direction

K  = [ ones(1,6) ;  ly.' ; -lx.' ; (P.k_m*sg).' ];   % [f;Mx;My;Mz] = K*T
Kd = pinv(K);
G  = Kd(:,2:4);          % rotor thrust deviation per unit [Mx My Mz]

%% ------------------- Thrust budget ----------------------------------
T_tot = P.mass_kgf*P.g;
Tbar  = T_tot/6;
T_min = P.C_T*P.rpm_min^2;
T_max = P.C_T*P.rpm_max^2;
up    = T_max - Tbar;                  % upward   headroom per rotor
dn    = Tbar  - T_min;                 % downward headroom per rotor
dmax  = min(up,dn);

gInf  = max(abs(G),[],1);              % ||G*e_j||_inf  for j = x,y,z
Mmax  = dmax ./ gInf;                  % clamp-onset capacity per axis

fprintf('T_tot = %.3f N  (%.3f kgf),  Tbar = %.4f N\n', T_tot, P.mass_kgf, Tbar);
fprintf('rotor bounds  : [%.4f, %.4f] N  ->  up = %.4f, dn = %.4f N\n', ...
        T_min, T_max, up, dn);
fprintf('||G e_j||_inf : Mx %.4f | My %.4f | Mz %.4f  N/(N*m)\n', gInf);
fprintf('capacity      : Mx %.4f | My %.4f | Mz %.4f  N*m\n', Mmax);
fprintf('yaw/roll cost ratio = %.2f\n\n', gInf(3)/gInf(1));

Mz_max = Mmax(3);

%% ================= Fig.1 : (Mx,My) slices over Mz ===================
Mz_list = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.28];
cmap    = parula(numel(Mz_list));

figure('Name','Admissible (Mx,My) set vs Mz','Color','w', ...
       'Position',[80 80 1180 500]);

subplot(1,2,1); hold on; grid on; axis equal
for k = 1:numel(Mz_list)
    V = momentPolygon(G, up, dn, Mz_list(k));
    if isempty(V), continue; end
    plot([V(:,1);V(1,1)], [V(:,2);V(1,2)], '-', 'LineWidth',1.8, ...
         'Color',cmap(k,:), 'DisplayName',sprintf('M_z = %.2f',Mz_list(k)));
end
xline(0,'Color',[.6 .6 .6]); yline(0,'Color',[.6 .6 .6]);
xlabel('M_x  [N\cdotm]'); ylabel('M_y  [N\cdotm]');
title(sprintf('Admissible roll/pitch set  (T_{tot} = %.2f N)', T_tot));
legend('Location','eastoutside','FontSize',8);

% ---- companion: rotor-thrust bar chart at one operating point --------
subplot(1,2,2); hold on; grid on
Mdemo = [1.0; 0.0; 0.15];                       % example command
dT    = G*Mdemo;
bar(1:6, Tbar + dT, 0.55, 'FaceColor',[.35 .55 .85]);
yline(T_max,'r--','T_{max}','LabelHorizontalAlignment','left');
yline(T_min,'r--','T_{min}','LabelHorizontalAlignment','left');
yline(Tbar ,'k:' ,'$\bar{T}$','Interpreter','latex');
xlabel('rotor index'); ylabel('T_i  [N]');
title(sprintf('rotor thrusts at M = [%.2f, %.2f, %.2f] N\\cdotm', Mdemo));
ylim([0 T_max*1.1]);

%% ================= Fig.2 : area / extent vs Mz ======================
Mz_scan = linspace(0, Mz_max*1.02, 240);
areaMz  = zeros(size(Mz_scan));
extX    = zeros(size(Mz_scan));
extY    = zeros(size(Mz_scan));
nVert   = zeros(size(Mz_scan));

for k = 1:numel(Mz_scan)
    V = momentPolygon(G, up, dn, Mz_scan(k));
    if isempty(V), continue; end
    areaMz(k) = polyarea(V(:,1), V(:,2));
    extX(k)   = max(abs(V(:,1)));
    extY(k)   = max(abs(V(:,2)));
    nVert(k)  = size(V,1);
end

figure('Name','Set size vs Mz','Color','w','Position',[80 620 1180 420]);

subplot(1,3,1); hold on; grid on
plot(Mz_scan, areaMz, 'LineWidth',1.8);
xline(Mz_max,'r--',sprintf('M_z^{max}=%.3f',Mz_max));
xlabel('M_z  [N\cdotm]'); ylabel('area of (M_x,M_y) set  [(N\cdotm)^2]');
title('set area shrinks with yaw demand');

subplot(1,3,2); hold on; grid on
plot(Mz_scan, extX, 'LineWidth',1.8, 'DisplayName','max |M_x|');
plot(Mz_scan, extY, 'LineWidth',1.8, 'DisplayName','max |M_y|');
% conservative budget prediction (aligned-sign, sufficient condition)
plot(Mz_scan, Mmax(1)*(1-Mz_scan/Mz_max), 'k--', 'DisplayName','budget bound M_x');
xline(Mz_max,'r--');
xlabel('M_z  [N\cdotm]'); ylabel('extent  [N\cdotm]');
title('per-axis extent'); legend('Location','southwest','FontSize',8);

subplot(1,3,3); hold on; grid on
stairs(Mz_scan, nVert, 'LineWidth',1.6);
xline(Mz_max,'r--');
ylim([0 8]); yticks(0:8);
xlabel('M_z  [N\cdotm]'); ylabel('number of active constraints');
title('hexagon (6) \rightarrow triangle (3)');

%% ================= Fig.3 : animation (optional) =====================
if ANIMATE
    figure('Name','Animation','Color','w');
    for Mz = linspace(0, Mz_max*0.99, 120)
        V = momentPolygon(G, up, dn, Mz);
        cla; hold on; grid on; axis equal
        axis([-3.6 3.6 -3.2 3.2]);
        if ~isempty(V)
            fill(V(:,1), V(:,2), [.4 .6 .9], 'FaceAlpha',0.35, 'EdgeColor','b','LineWidth',1.6);
        end
        xline(0,'Color',[.6 .6 .6]); yline(0,'Color',[.6 .6 .6]);
        xlabel('M_x  [N\cdotm]'); ylabel('M_y  [N\cdotm]');
        title(sprintf('M_z = %.3f N\\cdotm   (M_z^{max} = %.3f)', Mz, Mz_max));
        drawnow; pause(0.02);
    end
end

%% ===================== local functions ==============================
function V = momentPolygon(G, up, dn, Mz)
%MOMENTPOLYGON  Vertices of the feasible (Mx,My) polygon at a given Mz.
%   Exact vertex enumeration of  A*[Mx;My] <= b  (12 half-planes).
%   Returns [] when the set is empty (|Mz| beyond the yaw capacity).

    A = [  G(:,1:2) ; -G(:,1:2) ];
    b = [ up - G(:,3)*Mz ; dn + G(:,3)*Mz ];

    n = size(A,1);
    V = zeros(0,2);
    tol = 1e-7;

    for i = 1:n-1
        for j = i+1:n
            Am = A([i j],:);
            if abs(det(Am)) < 1e-9,  continue;  end     % parallel
            p = Am \ b([i j]);
            if all(A*p <= b + tol)                      % feasible vertex
                V(end+1,:) = p.';                       %#ok<AGROW>
            end
        end
    end

    if isempty(V), return; end

    V = uniquetol(V, 1e-6, 'ByRows', true);
    cen = mean(V,1);
    [~,idx] = sort(atan2(V(:,2)-cen(2), V(:,1)-cen(1)));  % CCW ordering
    V = V(idx,:);
end
