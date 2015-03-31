function [models,logP]=gwmcmc(minit,logPfuns,mccount,varargin)
%% Cascaded affine invariant ensemble MCMC sampler. "The MCMC hammer"
%
% GWMCMC is an implementation of the Goodman and Weare 2010 Affine
% invariant ensemble Markov Chain Monte Carlo (MCMC) sampler. MCMC sampling
% enables bayesian inference. The problem with many traditional MCMC samplers
% is that they can have slow convergence for badly scaled problems, and that
% it is difficult to optimize the random walk for high-dimensional problems.
% This is where the GW-algorithm really excels as it is affine invariant. It
% can achieve much better convergence on badly scaled problems. It is much
% simpler to get to work straight out of the box, and for that reason it
% truly deserves to be called the MCMC hammer.
%
% (This code uses a cascaded variant of the Goodman and Weare algorithm).
%
% USAGE:
%  [models,logP]=gwmcmc(minit,logPfuns,mccount,[Parameter,Value,Parameter,Value]);
%
% INPUTS:
%     minit: an MxW matrix of initial values for each of the walkers in the
%            ensemble. (M:number of model params. W: number of walkers). W
%            should be atleast 2xM. (see e.g. mvnrnd).
%  logPfuns: a cell of function handles returning the log probality of a
%            proposed set of model parameters. Typically this cell will
%            contain two function handles: one to the logprior and another
%            to the loglikelihood. E.g. {@(m)logprior(m) @(m)loglike(m)}
%   mccount: What is the desired total number of monte carlo proposals.
%            This is the total number, -NOT the number per chain.
%
% Named Parameter-Value pairs:
%   'StepSize': unit-less stepsize (default=2.5).
%   'ThinChain': Thin all the chains by only storing every N'th step (default=10)
%   'ProgressBar': Show a text progress bar (default=true)
%   'Parallel': Run in ensemble of walkers in parallel. (default=false)
%
% OUTPUTS:
%    models: A MxWxT matrix with the thinned markov chains (with T samples
%            per walker). T=~mccount/p.ThinChain/W.
%      logP: A PxWxT matrix of log probabilities for each model in the
%            models. here P is the number of functions in logPfuns.
%
% Note on cascaded evaluation of log probabilities:
% The logPfuns-argument can be specifed as a cell-array to allow a cascaded
% evaluation of the probabilities. The computationally cheapest function should be
% placed first in the cell (this will typically the prior). This allows the
% routine to avoid calculating the likelihood, if the proposed model can be
% rejected based on the prior alone.
% logPfuns={logprior loglike} is faster but equivalent to
% logPfuns={@(m)logprior(m)+loglike(m)}
%
% TIP: if you aim to analyze the entire set of ensemble members as a single
% sample from the distribution then you may collapse output models-matrix
% thus: models=models(:,:); This will reshape the MxWxT matrix into a
% Mx(W*T)-matrix while preserving the order.
%
%
% EXAMPLE:
% Here we sample a multivariate normal distribution.
% %define problem:
% mu = [5;-3;6];
% C = [.5 -.4 0;-.4 .5 0; 0 0 1];
% iC=pinv(C);
% logPfuns={@(m)-0.5*sum((m-mu)'*iC*(m-mu))}
%
% %make a set of starting points for the entire ensemble of walkers
% minit=randn(length(mu),length(mu)*2);
%
% %Apply the MCMC hammer
% [models,logP]=gwmcmc(minit,logPfuns,100000);
% models(:,:,1:floor(end/5))=[]; %remove 20% as burn-in
% models=models(:,:)'; %reshape matrix to collapse the ensemble member dimension
% scatter(models(:,1),models(:,2))
% prctile(models,[5 50 95])
%
%
% References:
% Goodman & Weare (2010), Ensemble Samplers With Affine Invariance, Comm. App. Math. Comp. Sci., Vol. 5, No. 1, 65–80
% Foreman-Mackey, Hogg, Lang, Goodman (2013), emcee: The MCMC Hammer, arXiv:1202.3665
%
% WebPage: https://github.com/grinsted/gwmcmc
%
% -Aslak Grinsted 2015


if nargin<3
    error('GWMCMC:toofewinputs','GWMCMC requires atleast 3 inputs.')
end
M=size(minit,1);
if size(minit,2)==1
    minit=bsxfun(@plus,minit,randn(M,M*5));
end


p=inputParser;
p.addParameter('StepSize',2.5,@isnumeric);
p.addParameter('ThinChain',10,@isnumeric);
p.addParameter('ProgressBar',true,@islogical);
p.addParameter('Parallel',false,@islogical);
p.parse(varargin{:});
p=p.Results;


Nwalkers=size(minit,2);

if size(minit,1)*2>size(minit,2)
    warning('GWMCMC:minitdimensions','Check minit dimensions.\nIt is recommended that there be atleast twice as many walkers in the ensemble as there are model dimension.')
end

if p.ProgressBar
    progress=@textprogress;
else
    progress=@noaction;
end


Nkeep=ceil(mccount/p.ThinChain/Nwalkers); %number of samples drawn from each walker
mccount=(Nkeep-1)*p.ThinChain+1;

models=nan(M,Nwalkers,Nkeep); %pre-allocate output matrix

models(:,:,1)=minit;

if ~iscell(logPfuns)
    logPfuns={logPfuns};
end

NPfun=numel(logPfuns);

%calculate logP state initial pos of walkers
logP=nan(NPfun,Nwalkers,Nkeep);
for wix=1:Nwalkers
    for fix=1:NPfun
        logP(fix,wix,1)=logPfuns{fix}(minit(:,wix));
    end
end
if ~all(all(isfinite(logP(:,:,1))))
    error('Starting points for all walkers must have finite logP')
end


reject=zeros(Nwalkers,1);


curm=models(:,:,1);
curlogP=logP(:,:,1);
progress(0,0,0)
totcount=Nwalkers;
for row=2:Nkeep
    for jj=1:p.ThinChain
        %generate proposals for all walkers
        %(done outside walker loop, in order to be compatible with parfor - some penalty for memory):
        %-Note it appears to give a slight performance boost for non-parallel.
        rix=mod((1:Nwalkers)+floor(rand*(Nwalkers-1)),Nwalkers)+1; %pick a random partner
        zz=((p.StepSize - 1)*rand(1,Nwalkers) + 1).^2/p.StepSize;
        proposedm=curm(:,rix) - bsxfun(@times,(curm(:,rix)-curm),zz);
        logrand=log(rand(NPfun+1,Nwalkers)); %moved outside because rand is slow inside parfor 
        if p.Parallel
            %parallel/non-parallel code is currently mirrored in
            %order to enable experimentation with separate optimization
            %techniques for each branch. 
            
            parfor wix=1:Nwalkers
                cp=curlogP(:,wix);
                lr=logrand(:,wix);
                acceptfullstep=true;
                proposedlogP=nan(NPfun,1);
                if lr(1)<(numel(proposedm(:,wix))-1)*log(zz(wix))
                    for fix=1:NPfun
                        proposedlogP(fix)=logPfuns{fix}(proposedm(:,wix)); %have tested workerobjwrapper but that is slower.
                        if lr(fix+1)>proposedlogP(fix)-cp(fix)
                            acceptfullstep=false;
                            break
                        end
                    end
                else
                    acceptfullstep=false;
                end
                if acceptfullstep
                    curm(:,wix)=proposedm(:,wix); curlogP(:,wix)=proposedlogP;
                else
                    reject(wix)=reject(wix)+1;
                end
            end
        else %NON-PARALLEL
            for wix=1:Nwalkers 
                acceptfullstep=true;
                proposedlogP=nan(NPfun,1);
                if logrand(1,wix)<(numel(proposedm(:,wix))-1)*log(zz(wix))
                    for fix=1:NPfun
                        proposedlogP(fix)=logPfuns{fix}(proposedm(:,wix));
                        if logrand(fix+1,wix)>proposedlogP(fix)-curlogP(fix,wix)
                            acceptfullstep=false;
                            break
                        end
                    end
                else
                    acceptfullstep=false;
                end
                if acceptfullstep
                    curm(:,wix)=proposedm(:,wix); curlogP(:,wix)=proposedlogP;
                else
                    reject(wix)=reject(wix)+1;
                end
            end
           
        end
        totcount=totcount+Nwalkers;        
        progress((row-1+jj/p.ThinChain)/Nkeep,curm,sum(reject)/totcount)
    end
    models(:,:,row)=curm;
    logP(:,:,row)=curlogP;
    
    %progress bar
    
end
progress(1,0,0);


% TODO: make standard diagnostics to give warnings...
% TODO: make some diagnostic plots if nargout==0;



function textprogress(pct,curm,rejectpct)
persistent lastNchar lasttime starttime
if isempty(lastNchar)||pct==0
    lastNchar=0;lasttime=cputime-10;starttime=cputime;fprintf('\n')
    pct=1e-16;
end
if pct==1
    fprintf('%s',repmat(char(8),1,lastNchar));lastNchar=0;
    return
end
if (cputime-lasttime>0.1)
    
    ETA=datestr((cputime-starttime)*(1-pct)/(pct*60*60*24),13);
    progressmsg=[183-uint8((1:40)<=(pct*40)).*(183-'¤') ''];
    %progressmsg=[uint8((1:40)<=(pct*40)).*'#' ''];
    curmtxt=sprintf('% 9.3g\n',curm(1:min(end,20),1));
    %curmtxt=mat2str(curm);
    progressmsg=sprintf('GWMCMC %5.1f%% [%s] %s\n%3.0f%% rejected\n%s\n',pct*100,progressmsg,ETA,rejectpct*100,curmtxt);
    
    fprintf('%s%s',repmat(char(8),1,lastNchar),progressmsg);
    drawnow;lasttime=cputime;
    lastNchar=length(progressmsg);
end

function noaction(varargin)

% Acknowledgements: I became aware of the GW algorithm via a student report
% which was using emcee for python. Great stuff.