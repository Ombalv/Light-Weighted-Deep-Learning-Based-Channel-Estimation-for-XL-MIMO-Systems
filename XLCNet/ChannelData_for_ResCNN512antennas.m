% XL-MIMO channel generation
clear
N=512; % BS antennas
lamda=0.01;
Anten_dis=lamda/2; % antenna space
Anten_dis_norm=0.5; % normalized antenna space
Lf=1; % number of paths for far-field part
Ln=5; % number of paths for near-field part

% training data size
% num_sta=1000;
% num_ffading=100;

% test data size
num_sta=10;
num_ffading=200;

num_train=num_sta*num_ffading;
Channel_mat=zeros(num_train,N);
Channel_near_mat=zeros(num_train,N);
Channel_far_mat=zeros(num_train,N);

for i=1:num_sta
    Cp_far=1; % channel average power of far part
    Cp_near=1; % channel average power of near part
    % angle_index_far=Allcomb_far(unidrnd(num_comb_far),:);
    % theta_far=angle_grid(angle_index_far); % AoA at BS of far part
    theta_far=pi*unifrnd(-1/2,1/2,1,Lf); % AoA at BS of far part
    % angle_index_near=Allcomb_near(unidrnd(num_comb_near),:);
    % theta_near=angle_grid(angle_index_near); % AoA at BS of near part
    theta_near=pi*unifrnd(-1/2,1/2,1,Ln); % AoA at BS of near part
    r_near=70*rand(1,Ln)+10;
    
    for n=1:num_ffading
        % generate far-field channel
        A_far=1/sqrt(N)*exp((-1j*2*pi*Anten_dis/lamda*[0:1:N-1]')*sin(theta_far));
        f_far=(normrnd(0,Cp_far/sqrt(2),Lf,1)+1j*normrnd(0,Cp_far/sqrt(2),Lf,1));
        h_far=A_far*f_far;

        % generate near-field channel
        delta=(2*[1:N]'-N-1)/2;
        B_near=zeros(N,Ln);
        for l=1:Ln
            r_dis=sqrt(r_near(l)^2+delta.^2*Anten_dis^2-2*r_near(l)*delta*Anten_dis*sin(theta_near(l)));
            B_near(:,l)=1/sqrt(N)*exp(-1j*2*pi/lamda*(r_dis-r_near(l)));
        end
        f_near=(normrnd(0,Cp_near/sqrt(2),Ln,1)+1j*normrnd(0,Cp_near/sqrt(2),Ln,1));
        h_near=B_near*f_near;
        h_hyb=sqrt(N/(Lf+Ln))*(h_far+h_near);        
        Channel_mat((i-1)*num_ffading+n,:)=h_hyb.';
        % Channel_far_mat((i-1)*num_ffading+n,:)=sqrt(N/(Lf+Ln))*h_far.';
        % Channel_near_mat((i-1)*num_ffading+n,:)=sqrt(N/(Lf+Ln))*h_near.';
        Channel_far_mat((i-1)*num_ffading+n,:)=sqrt(N/(Lf))*h_far.';
        Channel_near_mat((i-1)*num_ffading+n,:)=sqrt(N/(Ln))*h_near.';
    end
end

% save(['.\data\Channel_f1n5_512ANTS_1000by100'],'Channel_mat')

save(['.\data\Channel_f1n5_512ANTS_10by200'],'Channel_mat')

% save(['.\data\Channel_f1n5_far_256ANTS_10by200'],'Channel_far_mat')
% save(['.\data\Channel_f1n5_near_256ANTS_10by200'],'Channel_near_mat')

% save(['...\Channel_f1n5_256ANTS_10by200'],'Channel_mat')
% save(['.\data\Channel_f1n5_256ANTS_10by200'],'Channel_mat')
