
%Thanks to Arpad Rozsas for inspiration. 

%This demo is not finished yet. 



%Observed data: The weight of 30 rats measured for 5 weeks.
Y = [151 145 147 155 135 159 141 159 177 134 160 143 154 171 163 160 142 156 157 152 154 139 146 157 132 160 169 157 137 153
  199 199 214 200 188 210 189 201 236 182 208 188 200 221 216 207 187 203 212 203 205 190 191 211 185 207 216 205 180 200
  246 249 263 237 230 252 231 248 285 220 261 220 244 270 242 248 234 243 259 246 253 225 229 250 237 257 261 248 219 244
  283 293 312 272 280 298 275 297 350 260 313 273 289 326 281 288 280 283 307 286 298 267 272 285 286 303 295 289 258 286
  320 354 328 297 323 331 305 338 376 296 352 314 325 358 312 324 316 317 336 321 334 302 302 323 331 345 333 316 291 324]';
Nrat=size(Y,1);

%x = day of observation.
x       = [8, 15, 22, 29, 36];
x_bar   = 22;


%helper function same as log(normpdf):
lognormpdf=@(x,mu,sigma)-0.5*((x-mu)./sigma).^2  -log(sqrt(2*pi).*sigma);

alpha=@(m)m(5+(1:Nrat));
beta=@(m)m(35+(1:Nrat));

expectedweight=@(m)bsxfun(@plus,alpha(m),beta(m)*(x-x_bar));
%m1=alpha_c
%m2=alpha_sigma
%m3=beta_c
%m4=beta_sigma
%m5=sigma_c

logP = {  @(m)sum(lognormpdf(m(1:5),[median(Y(:)) 3 6 -.5 1]',[50 .5 4 .5 .5]'))  ...
          @(m)sum(lognormpdf(alpha(m),m(1),exp(m(2))))   ...
          @(m)sum(lognormpdf(beta(m),m(3),exp(m(4))))  ...
          @(m)sum(sum(lognormpdf(Y,expectedweight(m),exp(m(5)))))};


for ii=1:30,q(ii,:)=polyfit(x-x_bar,Y(ii,:),1);end
p=median(q);

alpha0=q(:,2);
beta0=q(:,1);
minit=[p(2);log(std(alpha0));p(1);log(std(beta0));log(1);alpha0;beta0];
%minit=fminsearch(@(m)-logP(m),minit)
M=length(minit);

minit=bplus(minit,randn(M,M*3)*1);

tic
m=gwmcmc(minit,logP,10000,1.02);
toc  
m=m(:,:)';

alpha0=m(:,1)-m(:,3)*x_bar;
mean(alpha0)
std(alpha0)
