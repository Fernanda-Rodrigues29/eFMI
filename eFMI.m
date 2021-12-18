function [ erro, ys, r, num_clusters] = eFMI(x, output, basic_alpha, window_size ,sigma_init, num_training_points, com_p, N_max, pas)

% Encontre dimensao e numero de pontos
[num_points, num_vars] = size(x);

% Encontre dimensao e numero de pontos
[num_points, num_vars] = size(x);
%Definição do parâmetro lamba
if window_size >= 100
    lambda=0.01;
elseif window_size <100 || window_size>=20
    lambda=0.05;
elseif window_size <20 
    lambda=0.1;
end

% Calcula Chi-Quadrado usado na equacao 3.7
chi_threshold1 = chi2inv(1-lambda, num_vars); %Limiar de alerta
Nr =window_size;
d_max1 = linspace(0,1,num_training_points);
for i1= 1:num_training_points
    d = max(x(i1,:)) - min(x(i1,:));
    d_max1(i1) = d/2*Nr;   
end
d_max = min(d_max1);
chi_threshold = exp((-d_max)^2); %Limiar de alerta

for i=1:num_points
    if i == 1
        % Inicializa variaveis
        K_init = sigma_init; %Matriz de dispersão incial
        alpha_init = basic_alpha; %Alfa incial
        c=1; %primeiro centro
        v(1:c,:) = x(1,:); %vetor dos centro de cluster
        K{1:c} = K_init;
        Kinv{1:c} = inv(K_init);
        cluster_num_points(1:c) = 0; %Qtd de cluster
        a(1:c,1:num_points) = 0; % Indice de alerta
        o(1:c,1:num_points) = 0; % Valor da ocorrencia - Equacao 3.8
        Q_init = 1000*eye(num_vars+1);
        Q1{c} = eye(pas); %Parâmetros do consequente
        Q{1:c} = Q_init;%Parâmetros do consequente
        gamma{1:c} = [output(1) zeros(1,num_vars)]';%Parâmetros do consequente
        active_centers = [1];
        num_clusters(1:num_points) = 1;
        Ie(1:c,1:num_points) = 0; % índice de etiqueta
        age(1:c,1:num_points) = 0;
        cont =0;
        omega_init = 1/10^2*ones(1,num_vars+1);
        omega{num_points,1:c} = [omega_init(1) zeros(1,num_points-1)];
        omega{1} = omega_init;
        Y1{num_points,1:c} = [zeros(1,num_points-1)]';
        Y1{1} = output(1);
        P_init = 10^2 * eye(num_vars+1);%Parâmetros do consequente
        P1{1:c} = P_init;
        step{1:c} = [0]';
        step{1}= 0;
    
    end
    num_clusters(i) = length(active_centers);
    for k=1:length(active_centers)
        j = active_centers(k);
        nc = cluster_num_points(j);
        
        if nc < N_max
            chistat(j,i) = (x(i,:)-v(j,:))*(x(i,:)-v(j,:))'; %Calculo da distância euclidiana
            p(j,i) = mvgaussmf(x(i,:),v(j,:),1); %Calculo de compatibilidade entre os grupos
            distancia = 1;
        else 
            chistat(j,i) = (x(i,:)-v(j,:))*Kinv{j}*(x(i,:)-v(j,:))'; %Calculo da distância Mahalanobis
            p(j,i) = mvgaussmf(x(i,:),v(j,:), Kinv{j}); %Calculo de compatibilidade entre os grupos
            distancia = 2;
        end
        %Indica uma violação do limiar de compatibilidade
        if distancia== 1
            if (p(j,i) > chi_threshold)
                o(j,i) = 1;
            end
        else
             if (chistat(j,i) > chi_threshold1)
                o(j,i) = 1;
             end
        end
        
        %Acrescento mais um ponto ao cluster
        cluster_num_points(j) = cluster_num_points(j) + 1;
        %Verifica se o ponto está fora da janela definida, caso esteja
        %adicionamos o ponto no vetor do  índice de alerta
        if (cluster_num_points(j) > window_size)
             num_ocurrences(j,i) = sum(o(j,i-window_size+1:i));
            if (num_ocurrences(j,i) == 0)
                a(j,i) = 0;
            else
                %calcula uma função de distribuição cumulativa binomial em cada um dos valores num_ocurrences -1 usando o 
                %número correspondente de tentativas window_size e a
                %probabilidade de sucesso de cada tentativa em lambda
                a(j,i) = binocdf(num_ocurrences(j,i)-1, window_size, lambda);
            end
        end
    end
    %Os antecedentes das regras 
    xk = [ 1 x(i,:)]';
    
    ys(i) = 0;%parâmetros do consequente incial
    sump = 0;
    
    ys1(i) = 0;
    for k=1:length(active_centers)
        j = active_centers(k);
        tem2 =step{j};
        yc(j) =xk'*gamma{j};%Parâmetros do consequente
        ys(i) = ys(i) +p(j,i)*yc(j);%Parâmetros do consequente
        sump = sump + p(j,i);
        sump1(j,i) = sump;
           % asass = omega{j}(tem2,:);
            %ys(i) = asass*gamma{j};%Parâmetros do consequente
           % yc(j) = ys(i);
 
    end
    
    
    %========================================================
        if (sump == 0)  %Se somatório da medida de compatibilidade
            ys(i) = ys(i-1); % A saída do modelo é igual a anterior
        else % Senão é calculada a média ponderadada saída de cada regra
             ys(i) = ys(i)/sump;
        end
		if (isnan(ys(i))==1) %Determinar quais elementos da matriz são NaN e Se onde os elementos de ys (i)são NaN
			i_ =i;
			yc_ = yc;
			sump_ = sump
			pause
		end
    %Saída do modelo


    r(i) = (output(i,:)-ys(i))^2; %  A diferença ao quadrado da saída dada para a saída do modelo
    erro(i) = (output(i,:)-ys(i)); % A diferença da saída dada para a saída do modelo
   
    %=========================================================
    
%     ys(i) = ys(i)/sump;
%     r(i) = (output(i,:)-ys(i))^2;
    
    %=========================================================
    
    
    %Se i for menor ou igual ao número de pontos do modelo
    if (i <= num_training_points)
        center = -1; %Definimos o center = -1
        compatibility = -inf;%Definimos o compatibility = -inf
        for k=1:length(active_centers)
            j = active_centers(k);
            %Se o o índice de alerta é menor que seu limiar ou se a
            %distância calculada é menor que Chi-Quadrado
            if distancia== 1
                if (a(j,i) < 1-lambda || p(j,i) < chi_threshold) %Limiar de alerta para criar nosso grupos é 1-lamba
                    %E se índice de compatibilidade é maior que -inf, então a
                    %compatibilidade recebe o valor do indice de
                    %compatibilidade e o center é igual ao centro ativo
                    if (p(j,i) > compatibility)
                        compatibility = p(j,i);
                        center = j;
                    end
                end
            else
                if (a(j,i) < 1-lambda || chistat(j,i) < chi_threshold1) %Limiar de alerta para criar nosso grupos é 1-lamba
                %E se índice de compatibilidade é maior que -inf, então a
                %compatibilidade recebe o valor do indice de
                %compatibilidade e o center é igual ao centro ativo
                    if (p(j,i) > compatibility)
                        compatibility = p(j,i);
                        center = j;
                    end
                end
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%% CRIANDO UM GRUPO %%%%%%%%%%%%%%%%%%%%%
        % Se o center = -1, acrecenta um novo centro e center = qtd de
        % centro e cria um  novo grupo
        if (center == -1)
            c = c + 1; % Aumenta a qtd de cluster
            center = c; %Center é o centro do novo grupo 
            a(c,1:num_points) = 0; %Cria o índice de alerta para o grupo
            Q{c} = eye(num_vars+1);
            Q1{c} = eye(pas);
            P1{c} = P_init;
            step{c} = 1; %Acrscentando mais um elemento ao grupo
            omega{c} = omega_init;
            Y1{c} =  zeros(1,num_points)';
            Y1{c}(1)= output(i,:);
            step{c}(1,1) = 1;
            gamma{c} = zeros(1,num_vars+1)'; %Parâmetros do consequente
            gamma{c}(1) = output(i,:);%Parâmetros do consequente
            v(c,:) = x(i,:);%Acrescenta ao vetor dos centro de cluster os valores de entrada do novo centro
            K{c} = K_init; % Valor do novo centro na Matriz de disperção inicial
            Kinv{c} = inv(K_init);% Valor do novo centro na Matriz inversa de disperção inicial
            o(c,1:num_points) = 0;  % Valor da ocorrencia - Equacao 3.8
            cluster_num_points(c) = 0;% Qtd de pontos no cluster novo
            fprintf('%d - New cluster found\n', i);
            active_centers = union(active_centers,c);%Acrescenta ao vetor centros ativos o novo centro 
            nc1 = cluster_num_points(c);
            %Calcula o índice de compatibilidade da amostra com o novo centro
            if nc1 < N_max
                p(c,i) = mvgaussmf(x(i,:),v(c,:), 1);
            else
                p(c,i) = mvgaussmf(x(i,:),v(c,:), Kinv{c});
            end 
            sump1(c,i) = p(c,i) + sump;
            fprintf('Num clusters: %d\n',length(active_centers));
            centers = v(active_centers,:); %Matriz dos centros ativos
            class(i) = center;
            t(c) = i;
         
        %Senão ele atualiza os parâmetros
        

        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%  Atualiza os parâmetos de j %%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        alpha(center,i) = alpha_init*(p(center,i)^(1-a(i))); %Ajuste para o centro de maior grau de compatibilidade
        K{center} = (1-alpha(center,i)).*(K{center} + alpha(center,i).*((x(i,:)-v(center,:))'*(x(i,:)-v(center,:))));
        w = Kinv{center}*(x(i,:)-v(center,:))';
        Kinv{center} = (1/(1-alpha(center,i)))*(Kinv{center} - alpha(center,i)*((w*w')/(1+alpha(center,i)*w'*(x(i,:)-v(center,:))')));
        v(center,:) = v(center,:) + alpha(center,i)*(x(i,:)-v(center,:)); %Soma o vetor j dos centro de cluster + Ajuste para o centro de maior grau de compatibilidade
        
        %%%%%%%%%%%%%%%%%%%% Checa por grupos redundantes  %%%%%%%%%%%%%%%%%%%%
        j = center;
        for k=1:length(active_centers)
            % Se o centro que estamos atualizando for diferente do
            % centro comparado e o centro comparado e o que será atualizado for menor ou igual que o
            % número de centro ativos
            if (j ~=  k && k <= length(active_centers) && j <= length(active_centers))
                l = active_centers(j);
                m = active_centers(k);
                % Calcula a medida de compatibilidade do centro que
                % será atualizado e do centro comparado e o inverso tbm
                nc2 = cluster_num_points(j);
                nc3 = cluster_num_points(k);
                nc4 = nc3 + nc2;
                if nc4 < N_max
                    h1 = (v(m,:)-v(l,:))*1*(v(m,:)-v(l,:))';
                    h2 = (v(l,:)-v(m,:))*1*(v(l,:)-v(m,:))';
                    chi_threshold = chi_threshold;
                else
                    h1 = (v(m,:)-v(l,:))*Kinv{l}*(v(m,:)-v(l,:))';
                    h2 = (v(l,:)-v(m,:))*Kinv{m}*(v(l,:)-v(m,:))';
                    chi_threshold = chi_threshold1;
                end  
                nc2 = cluster_num_points(j);
                nc3 = cluster_num_points(k);
                nc4 = nc3 + nc2;
                %Se uma das medidas forem menor que o limiar de
                %compatibilidade eles poderão mesclar
                if (norm(v(m,:)-v(l,:)) <= com_p)
                    fprintf('%d - Clusters %d and %d have same mean\n',i, j,k);
                    v(l,:) = v(m,:) - nc3/nc4 * h1;%Calcula a média do centro que será atualizado e do centro comparado

                                            % FUNCAO A SER IMPLEMENTADA
                                            %K{l} = uni_matriz(K{l},K{m},v(l,:),v(m,:));

                                            % A SER SUBSTITUIDO
                    K{l} = uni_matriz2(K{l},K{m}); % Matriz de disperção inicial
                    omega{l} = [omega{l}; omega{m}];
                    tem2=step{l};
                    tem3 = step{m};
                    step{l}(1,1) = tem2 + tem3; %Acrscentando mais um elemento ao grupo
                    Q{c} = Q_init;%Atualização dos parâmetros do consequente
                    P1{c} = P_init;
                    Q1{c} = eye(pas); 
                    Y1{l} =  [Y1{l}(1:tem2); Y1{m}(1:tem3)];
                    gamma{l} = (p(l,i)*gamma{l} + p(m,i)*gamma{m})/(p(l,i)+p(m,i));%Atualização dos parâmetros do consequente pag.38
                   
                    active_centers = setdiff(active_centers,m); %Retira o valor de m no vetor de centros ativos
                    class(find(class==m)) = l; %Encontre índices no class é igual a m e substitui por 1
                    ret = Ie(m);
                    E_Ie(m) = m;
                    EX_Ie = E_Ie;
                    EX_Ie(EX_Ie== 0) =[];
                    Fe = Ie;
                    Fe = setdiff(Fe,ret,'stable');
                    Ie(m) = 0;
                    cont = 1;
                    Fe(Fe == 0) =[];
                    fprintf('Num clusters: %d\n',length(active_centers));
                    centers = v(active_centers,:);
                    c=c-1; % diminuir um centro de cluster 
                    
                end
            end
        
        end
   
        %Exclusão
        %Atualizar idade
        Tg = cluster_num_points(j);
        if Tg == 1
            age(j) = i ;
        else
            age(j) = i - Ie(j);
        end
        M_i = length(active_centers);
        if M_i >1
            [b_i, ind] = max(Fe);
            j1 = active_centers(ind);
            N_i= b_i;
            R = N_i/i; 
            if (R < 0.01) && age(j1)> 50 && M_i > 2
                 active_centers = setdiff(active_centers,j1); %Retira o valor de m no vetor de centros ativos
                 class(find(class==j1)) = 1; %Encontre índices no class é igual a m e substitui por 1
                 fprintf('Num clusters: %d\n',length(active_centers));
                 centers = v(active_centers,:)
                 c=c-1; % diminuir um centro de cluster
                 Ie(j) = 0;
                 Fe = Ie;
                 Fe(Fe == 0) =[];

            end
        end
        
        if cont == 0
            Fe = Ie;
            Ie(j) = i;

        else 
            for k1=1:length(active_centers)
                j2 =  active_centers(k1);
                if (j == j2)
                    Fe(k1) = i;
                end
            end
            yy = 0;
            for k2 = 1:length(EX_Ie);
                k3= EX_Ie(k2);
                if j== k3
                   yy = 1;
                   break;
                end
            end
            if yy == 0
                Ie(j) = i;
            end
        end
        %Atualização de parâmetros do consequentes
        
        for k=1:length(active_centers)
            j = active_centers(k);
            if cluster_num_points(j) ~= 0
                step{j}(1,1) = step{j}(1,1) + 1;
            end
            %Passo 3 para atualização
            
            vetor = [1 x(i,:)]; %valor dos dados de entrada e saída coletados da amostra i no grupo j
            tem1=step{j}(1,1);%quantos elementos tem no grupo
            omega{j}(tem1,:) = vetor;
            [tem3 ggg3]=size(omega{j});%o vetor varphi com os dados de entrada e saída coletados
            Y1{j}(tem1,:) = output(i); % Vetor de saída com os dados coletados
            d11=0;
            if tem1 > pas %Se qtd de elementos de grupo for maior que pas, então iremos calcular o erro anterior
                for k1=(tem1- pas + 1):tem1-1 %tamanho do vetor será o valor de pas
                    d11 = d11 + 1; 
                    ga(d11,:) = omega{j}(k1,:); %vetor com tamanho pas de entradas e saídas coletadas
                    gb(d11,1) = Y1{j}(k1); % vetor com tamanho pas de saídas coletadas
                end
            else %Se qtd de elementos de grupo for menor que pas, então não iremos calcular o erro anterior
                for k1=tem1:tem1  % Vetor com tamanho 1
                    d11 = tem1;
                    for k2= 1:(num_vars + 1)
                        ga(d11,k2) = omega{j}(k1,k2); %vetor do grupo da amostra tem1 de entradas e saídas coletadas
                    end
                    gb (tem1,1) = Y1{j}(k1); %vetor do grupo da amostra tem1 das saídas coletadas
                    if tem1 ==1
                        for k2= (tem1+1):ggg3
                            for k3 = (tem3+1):pas
                                ga(k3,k2) = 0;
                            end
                            for k3 = 2:pas
                                gb (k3, 1) = 0;
                            end
                        end
                    end
                end
                

            end
            
            tem2=step{j}(1,1);
            d11=0;
            if tem2>pas
                for k1=(tem2- pas + 1):tem2
                    d11 = d11 + 1; 
                    Q1{j}(d11,d11)= p(j,i)/sump1(j,i);%Parâmetros do consequente
                end
            else
                for k1=tem2:tem2
                    d11 = d11 + 1; 
                    Q1{j}(d11,d11)= p(j,i)/sump1(j,i); %Parâmetros do consequente
                end
            end
        
            
            %Passo 4  para atualização        
            if tem1 > pas
                c112= [Q1{j} + (ga*P1{j})*ga'];
                L{j}= ([P1{j}*ga']*inv(c112)); %Calcular L(i)
                P1{j}= P1{j} - P1{j}*(ga'*L{j}');%Calcular P(i)
                dd2 = ga*gamma{j};
                dd3= gb;
                dd = [dd3 - dd2]';
                dd4 = L{j}*dd';
                gamma{j} = (gamma{j} + dd4); %Calcula o parâmetro do consequente(omega)
            else
                c112= [Q1{j} + (ga*P1{j})*ga'];
                L{j}= ([P1{j}*ga']*inv(c112)); %Calcular L(i)
                P1{j}= P1{j} - P1{j}*(ga'*L{j}');%Calcular P(i)
                dd2 =  ga*gamma{j};
                dd3 = gb;
                dd = [dd3 - dd2];
                dd4 = L{j}*dd;
                gamma{j} = (gamma{j} + dd4); %Calcula o parâmetro do consequente(omega)
             end
         
            
            %%Atualização dos parâmetros do consequente pag.37
            %gamma{j} = gamma{j} + Q{j}*xk*p(j,i)*(output(i,:)-xk'*gamma{j});%Atualização dos parâmetros do consequente
        end


    end
end



if num_vars == 2
	figure;
	hold on;
	for k=1:length(x)
		plot(x(k,1),x(k,2),'k.');
	end
	for k=1:length(active_centers)
		plot(v(active_centers(k),1), v(active_centers(k),2),'b.', 'MarkerSize',15);
		% FUNCAO QUE PLOTA O CONTORNO DA ELIPSE DE ERRO DADA A MATRIZ DE DISPERSAO
		error_ellipse(K{active_centers(k)},v(active_centers(k),:),'style','b--','conf',1-lambda);
	end
	xlabel('A')
	ylabel('B');
end
    
%adicionado em abril de 2019 - plota as funcoes
for k=1:length(active_centers)
	l = active_centers(k);
	fprintf('%d - IF x is [',k);
	for j=1:num_vars
		fprintf('%.4f ', v(l,j));
	end
	fprintf('] THEN y = %.4f', gamma{l}(1)); 
	for j=2:num_vars+1
		if (gamma{l}(j) >= 0)
			sig = '+';
		else
			sig = '-';
		end
		fprintf(' %c %2.4f x%d',sig, abs(gamma{l}(j)),j-1);
	end
	fprintf('\n');
end
%% Plota os pontos e os clusters (contorno)
if num_vars == 2
	figure;
	hold on;
    a= length(x);
	for k=1:length(x)
		plot(x(k,1),x(k,2),'k.');
	end
	for k=1:length(active_centers)
		plot(v(active_centers(k),1), v(active_centers(k),2),'b.', 'MarkerSize',15);
		% FUNCAO QUE PLOTA O CONTORNO DA ELIPSE DE ERRO DADA A MATRIZ DE DISPERSAO
		error_ellipse(K{active_centers(k)},v(active_centers(k),:),'style','b--','conf',1-lambda);
	end
	xlabel('Amostras')
	ylabel('y');
end


end
