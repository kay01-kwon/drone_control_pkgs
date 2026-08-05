%% plot_moment_set.m
%  Publication figure: admissible (Mx, My) moment set of a co-planar
%  hexarotor at FIXED total thrust, for several commanded yaw moments Mz.
%
%  Allocator model : pseudo-inverse followed by per-rotor clamping
%       T_i = Tbar + dT_i ,   dT = G*[Mx;My;Mz] ,   G = pinv(K)(:,2:4)
%       -dn <= dT_i <= up      (per-rotor box constraint)
%  For fixed Mz the feasible (Mx,My) region is the polytope
%       A*[Mx;My] <= b(Mz),  A = [G(:,1:2); -G(:,1:2)],
%       b = [up - G(:,3)*Mz ; dn + G(:,3)*Mz]
%  obtained here by exact vertex enumeration (no gridding).
%
%  Dotted circles: origin-centred inscribed radius
%       r_in(Mz) = Mx_max * (1 - |Mz|/Mz_max)
%  = roll/pitch authority guaranteed in EVERY direction.  The polytopes
%  themselves are not nested (opposed-sign yaw and roll partially cancel
%  on the most-loaded rotor), so r_in is the meaningful measure.
%
%  Output : moment_set.png at 600 dpi
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

Mz_list = [0.00, 0.10, 0.20, 0.30];      % commanded yaw moments [N*m]
OUT_PNG = 'moment_set.png';
DPI     = 600;

%% ------------------- Allocation matrix ------------------------------
c = cos(pi/3);  s = sin(pi/3);
ly = P.L*[ c;  1;  c; -c; -1; -c];
lx = P.L*[ s;  0; -s; -s;  0;  s];
sg = [-1; 1; -1; 1; -1; 1];

K  = [ ones(1,6) ; ly.' ; -lx.' ; (P.k_m*sg).' ];   % [f;Mx;My;Mz] = K*T
G  = pinv(K);  G = G(:,2:4);

%% ------------------- Thrust budget ----------------------------------
T_tot = P.mass_kgf*P.g;
Tbar  = T_tot/6;
T_min = P.C_T*P.rpm_min^2;
T_max = P.C_T*P.rpm_max^2;
up    = T_max - Tbar;
dn    = Tbar  - T_min;
dmax  = min(up,dn);

gInf   = max(abs(G),[],1);
Mmax   = dmax ./ gInf;          % [Mx_max, My_max, Mz_max]
Mz_max = Mmax(3);

fprintf('T_tot  = %.3f N   Tbar = %.4f N   (up %.4f / dn %.4f)\n', ...
        T_tot, Tbar, up, dn);
fprintf('capacity: Mx %.4f | My %.4f | Mz %.4f  N*m\n', Mmax);
fprintf('yaw/roll cost ratio = %.2f\n', gInf(3)/gInf(1));

%% ------------------------- Figure -----------------------------------
% wider canvas so the outside legend does not squeeze the axes
fig = figure('Color','w','Units','centimeters','Position',[2 2 19 11]);
ax  = axes(fig); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
axis(ax,'equal');
set(ax,'FontSize',13,'GridAlpha',0.15,'Layer','top');

cmap = lines(numel(Mz_list));
th   = linspace(0,2*pi,361);

for k = 1:numel(Mz_list)
    Mz = Mz_list(k);
    V  = momentPolygon(G, up, dn, Mz);
    if isempty(V), continue; end

    r_in = Mmax(1)*(1 - abs(Mz)/Mz_max);

    % exact polytope
    plot(ax, [V(:,1);V(1,1)], [V(:,2);V(1,2)], '-', ...
         'LineWidth',1.8, 'Color',cmap(k,:), ...
         'DisplayName', sprintf('$M_z=%.2f$,\\ $r_{\\mathrm{in}}=%.2f$', Mz, r_in));

    % guaranteed authority (inscribed circle)
    plot(ax, r_in*cos(th), r_in*sin(th), ':', ...
         'LineWidth',1.0, 'Color',cmap(k,:), 'HandleVisibility','off');
end

xline(ax,0,'Color',[.6 .6 .6],'HandleVisibility','off');
yline(ax,0,'Color',[.6 .6 .6],'HandleVisibility','off');

xlabel(ax,'$M_x$ [N$\cdot$m]','Interpreter','latex','FontSize',15);
ylabel(ax,'$M_y$ [N$\cdot$m]','Interpreter','latex','FontSize',15);
title(ax, sprintf('$T_{\\mathrm{tot}} = %.2f$ N', T_tot), ...
      'Interpreter','latex','FontSize',15);

lg = legend(ax,'Interpreter','latex','FontSize',13,'Location','bestoutside');
lg.ItemTokenSize = [22 11];

xlim(ax,[-3.7 3.0]); ylim(ax,[-3.2 3.2]);

%% ------------------------- Export -----------------------------------
if exist('exportgraphics','file')
    exportgraphics(fig, OUT_PNG, 'Resolution', DPI, 'BackgroundColor','white');
else                                       % pre-R2020a fallback
    set(fig,'PaperPositionMode','auto');
    print(fig, OUT_PNG, '-dpng', sprintf('-r%d',DPI));
end
fprintf('saved %s at %d dpi\n', OUT_PNG, DPI);

%% ===================== local function ===============================
function V = momentPolygon(G, up, dn, Mz)
%MOMENTPOLYGON  Vertices (CCW) of the feasible (Mx,My) polygon at given Mz.
%   Exact vertex enumeration of A*[Mx;My] <= b over all constraint pairs.
%   Returns [] if the set is empty (|Mz| beyond the yaw capacity).

    A = [  G(:,1:2) ; -G(:,1:2) ];
    b = [ up - G(:,3)*Mz ; dn + G(:,3)*Mz ];

    n   = size(A,1);
    V   = zeros(0,2);
    tol = 1e-9;

    for i = 1:n-1
        for j = i+1:n
            Am = A([i j],:);
            if abs(det(Am)) < 1e-12,  continue;  end    % parallel rows
            p = Am \ b([i j]);
            if all(A*p <= b + tol)
                V(end+1,:) = p.';                       %#ok<AGROW>
            end
        end
    end

    if isempty(V), return; end

    V   = uniquetol(V, 1e-9, 'ByRows', true);
    cen = mean(V,1);
    [~,idx] = sort(atan2(V(:,2)-cen(2), V(:,1)-cen(1)));
    V   = V(idx,:);
end
