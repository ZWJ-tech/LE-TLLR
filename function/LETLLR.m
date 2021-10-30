function [model_LETLLR] = LETLLR( X, Y, optmParameter)
% Alternating iterative solution W¡¢A¡¢Q  ||XW-YA||F2+0.5*alphatrace(AW'W)+0.5*beta||W||L1+theta||Q||L2,1
   %% optimization parameters
    alpha            = optmParameter.alpha;
    gamma            = optmParameter.gamma;
    theta            = optmParameter.theta;
    beta             = optmParameter.beta;

    maxIter          = optmParameter.maxIter;
    miniLossMargin   = optmParameter.miniLossMargin;
%     miniB2LossMargin = optmParameter.miniB2LossMargin;
%     epsilon          = optmParameter.epsilon;
   %% initializtion
    num_dim= size(X,2);
    num_N=size(X,1);
    num_D=size(X,2);
    num_labels=size(Y,2);
%     num_labels=size(Y,2);
%     N_s_1 = N_s;
%     C_s_1 = C_s;
    XTX = X'*X;
    XTY = X'*Y;
    
    W_s   = (XTX + theta*eye(num_dim)) \ (XTY);
    W_s_1 = W_s;
    norm_1=norm(XTX)^2; 
%     N_s = (eye(num_N).*(X*X'))^(-0.5)*((eye(1,num_N)'*eye(1,num_N))-eye(num_N).*(X*X'))*(eye(num_N).*(X*X'))^(-0.5);
    N_s= eye(num_N,num_N);
    N_s_1=N_s;
    
%     C_s=(eye(num_labels).*(Y'*Y))^(-0.5)*((eye(1,num_labels)'*eye(1,num_labels))-eye(num_labels).*(Y'*Y))*(eye(num_labels).*(Y'*Y))^(-0.5);
    C_s=eye(num_labels,num_labels);
    C_s_1=C_s;
   
    U_s=eye(size(Y,1),num_D);
    U_s_1=U_s;
    
    V_s=eye((num_D),size(Y,2));
    V_s_1=V_s; 

%     U_s_1 = U_s;
%     V_s_1 = V_s;
    iter    = 1;
    bk = 1;
    bk_1 = 1; 
    oldloss=0;
%     partB2oldloss=0;

    
% optmParameter.tooloptions.maxiter = 60;
% optmParameter.tooloptions.gradnorm = 1e-5;
% param.tooloptions.stopfun = @mystopfun;
   %% proximal gradient
    while iter <= maxIter
    %%fix N,U,V and C solve W  

       XTNYC = X'*N_s_1*Y*C_s_1;
       Lip = sqrt(2*norm_1 + (norm(X'*N_s_1'*X)^2));
        
       W_s_k  = W_s + (bk_1 - 1)/bk * (W_s - W_s_1);
       
       Gw_s_k = W_s_k - 1/Lip * ((- XTY) - XTNYC + X'*N_s_1'*X*W_s_k + XTX*W_s_k*C_s_1);
       bk_1   = bk;
       bk     = (1 + sqrt(4*bk^2 + 1))/2;
       W_s_1 = W_s;
       W_s  = softthres(Gw_s_k,alpha/Lip);
       
       
 %fix C,U,V update N
       N_s_k = (X*W_s*C_s_1'*Y'+X*W_s*(W_s')*X'+U_s_1*V_s_1*C_s_1'*Y')*pinv(2*Y*C_s_1*(C_s_1')*Y'+X*W_s*(W_s')*X'+eye(num_N));
       N_s_1 = abs(N_s_k);
       N_s_1(logical(eye(size(N_s_1))))=0;
       
       
       %fix N,U,V update C  
       C_s_k = pinv(2*Y'*(N_s_1')*N_s_1*Y+(W_s')*(X')*X*W_s + eye(num_labels))*(Y'*(N_s_1')*X*W_s+(W_s')*(X')*X*W_s+Y'*(N_s_1')*U_s_1*V_s_1);
       C_s_1 = abs(C_s_k);
       C_s_1(logical(eye(size(C_s_1))))=0;

       %fix N,C,V update U
       U_s_k = N_s_1*Y*C_s_1*(V_s_1')*pinv(V_s_1*(V_s_1')+beta*eye(num_D));
       U_s_1=U_s_k;
       %fix N,C,U update V 
       V_s_1=pinv(U_s_1'*U_s_1+theta*eye(num_D))*(U_s_1')*N_s_1*Y*C_s_1;
       lg= sqrt((norm((U_s_1')*U_s_1)^2));
       for i=1:num_labels
             if norm(V_s_1(:,i),2)>gamma/lg
                  V_s_k(:,i)=softthres((norm(V_s_1(:,i),2)-gamma/lg),0)/((norm(V_s_1(:,i),2)))*V_s_1(:,i);
             else
                  V_s_k(:,i)=zeros(num_D,1);
             end
       end
                    V_s_1=V_s_k;


        partA = 0.5*norm((X*W_s-Y),'fro')^2; 
        partB = 0.5*norm((N_s_1*Y*C_s_1-X*W_s),'fro')^2 + 0.5*norm((N_s_1*X*W_s-X*W_s),'fro')^2 + 0.5*norm((X*W_s*C_s_1-X*W_s),'fro')^2 ;
        partC = 0.5*norm((N_s_1*Y*C_s_1-U_s_1*V_s_1),'fro')^2 + beta*(norm((U_s_1),'fro')^2+norm((N_s_1),'fro')^2+norm((C_s_1),'fro')^2);
        sparsity = sum(sum(W_s~=0));
%         predictionLoss = trace((X*W_s_1 - Y*A_s_1)'*(X*W_s_1 - Y*A_s_1))+0.5*alpha*trace(U_s_1*W_s_1'*W_s_1)+rho_s_1*norm((Y-Y*A_s_1-V_s_1+miu1_s_1/rho_s_1),'fro')^2;
        E21= sum(sqrt(sum(V_s_1.*V_s_1,2)),1);
%         ParameterLoss = (-1/(2*rho_s_1))*norm((miu1_s_1),'fro')^2+(1/(2*rho_s_1))*norm((A_s_1-U_s_1+miu2_s_1/rho_s_1),'fro')^2-(1/(2*rho_s_1))*norm((miu2_s_1),'fro')^2;
        totalloss = partA + partB + partC + gamma*E21 + alpha*sparsity;

        if abs(oldloss - totalloss) <= miniLossMargin

             break;
        elseif totalloss <=0

             break;
        else
             oldloss = totalloss;
        end
        

       iter=iter+1; 
    end

    model_LETLLR.W=W_s;
    model_LETLLR.N=N_s_1;
    model_LETLLR.C=C_s_1;
    model_LETLLR.U=U_s_1;
    model_LETLLR.V=V_s_1;
end

%% soft thresholding operator
function W = softthres(W_t,lambda)
    W = max(W_t-lambda,0) - max(-W_t-lambda,0); 
end
