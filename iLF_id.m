function [id,model,fit] = iLF_id(FRF,si,nn)
%% [id,model,fit] = iLF_id(FRF,si,nn)

% Please cite the works under "References" when using this program.

% The program computes the modal properties, the state space matrices, and
% the fitted model of the given FRFs via the Loewner Framework. The state
% space approximation has the form: H(s)=C(sE-A)^(-1)B.
%
%     Given:
%     FRF = set of FRF of a MIMO system given in the matrix format
%           m by n or n by m, where n>m and they are respectively the
%           frequency data points and the number of channels;
%           FRF(output,input,frequency index)
%     si  = the vector of the frequency data points, either n by 1 or 1 by
%           n;
%     nn  = the order of the Loewner Framework, nn can take the values of
%             both scalar and vector;
%
%     Returns:
%     id:    id.ident: a 2+m by n matrix where the first line are the
%                      natural frequency, the second line is the damping
%                      ratio, and the remaining represent the identified
%                      mode shape;
%            id.k:     the order of the LF
%     model: an indexable variable such that the corresponding state space
%            matrices can be accessed in such a fashion: A=model.A and
%            similarly for B, C, and D;
%     fit:   The resulting H(s) approximation for the values supplied in si
%
%      The abovementioned output take the form of an array structure when
%      nn is a vector.

%% Disclaimer
% This program is free software: you can redistribute it and/or modify  it
% under the terms of the GNU Lesser General Public License as published by
% the Free Software Foundation, either version 3 of the License, or any
% later version.
%
% This program is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
% General Public License for more details.
%
% You should have received a copy of the GNU General Public License and GNU
% Lesser General Public License along with this program. If not, see
% <http://www.gnu.org/licenses/>.

%% Credits
% Original LF for fitting of FRF for MIMO systems by:
% Antonio Cosmin Ionita
% Rice University
% 2013/05/06
% (http://aci.rice.edu/system-identification/)
% Implementation of Modal Properties Extraction in SIMO systems:
% Gabriele Dessena
% gabriele.dessena@cranfield.ac.uk
% Cranfield University
% 15/09/2021
%
% Update: Extension to MIMO systems
% Gabriele Dessena
% gdessena@ing.uc3m.es
% Universidad Carlos III de Madrid
% 2024/08/07
% 
% Please cite the works under "References" when using this program
%
%% Changelog
% 25/09/2022 - Ensured condition for correct SIMO fitting
% 07/08/2024 - Extended to MIMO systems
%% References
% [1] G. Dessena, M. Civera, Improved Tangential Interpolation-based 
%      Multi-input Multi-output Modal Analysis of a Full Aircrafts, 
%      Preprint, (2024). Available at: https://arxiv.org/abs/2408.03810
%      (DOI: 10.48550/arXiv.2408.03810)
% [2] G. Dessena, A tutorial for the improved Loewner Framework for modal 
%     analysis, Universidad Carlos III de Madrid,(2024).
%     (DOI: -)
% [3] G. Dessena, M. Civera, L. Zanotti Fragonara, D. I. Ignatyev, J. F. 
%     Whidborne, A Loewner-based system identification and structural 
%     health monitoring approach for mechanical systems, Structural 
%     Control and Health Monitoring, Vol. 2023 (2023). 
%     (DOI: 10.1016/j.laa.2007.03.008)

%% Verify input dimensions
if max(size(size(FRF)))==3
    mi = 1;
    pi = 2;
    ni = 3;
    if mi==pi || mi==ni || ni==pi
        error("Check your inputs! Same axis selected for multiple data")
    end
    if mi~=1 || pi~=2 || ni~=3
        for k = 1:size(FRF,ni)
            ii(ni) = k;
            for j = 1:size(FRF,pi)
                ii(pi) = j;
                for i = 1:size(FRF,mi)
                    ii(mi) = i;
                    Hi(i,j,k) = FRF(ii(i),ii(2),ii(3));
                end
            end
        end
    else
        Hi=FRF;
    end
elseif max(size(size(FRF)))==2
    if size(FRF,1)>size(FRF,2)
        FRF = FRF.';
    end
    Hi = reshape(FRF,[size(FRF,1) 1 size(FRF,2)]);
else
    error("Check the format of your FRF")
end

if size(si,1)<size(si,2)
    si=si.';
end

if sum(any(imag(si)))==0
    si = si.*complex(0,1);
end

%% Loewner matrix LL, shifted-Loewner matrix sLL
% use the 'real' option to return Loewner matrices that have real entries
[LL,sLL,mu,la,V,W] = iLF(si,Hi,'real');
[Y,svLL, X] = svd(LL);
% svLL = diag(svLL);

iter_array =sort(nn,'descend');
r_iter = 1:length(iter_array);
for ij = r_iter
    k=iter_array(ij);
    % project the Loewner matrices to obtain the reduced model of order k with
    % the state-space realization E, A, B, C
    Yk = Y(:,1:k);
    Xk = X(:,1:k);

    % state-space Loewner model G(s) = C*( (s*E-A)\B )
    E = Yk'*LL*Xk;
    A = Yk'*sLL*Xk;
    B = Yk'*V;
    C = -W*Xk;

    % if there are unstable poles, flip the sign of their real part to obtain a
    % stable model
    [Z,ew] = eig(A,E);
    idx = real(diag(ew)) > 0;
    % if there are unstable poles, get rid of E, and then diagonalize
    if any(idx)
        A = E\A;
        B = E\B;
        E = eye(k);
        % diagonalize
        [Z,ew] = eig(A);
        idx = real(diag(ew)) > 0;
        ew(idx,idx) = -real(ew(idx,idx))+1j*imag(ew(idx,idx));
        % A is diagonal, and C(:,i)*B(i,:) is the i-th residue matrix
        A = ew;
        B = Z\B;
        C = C*Z;
    end

    D = 0; %set D matrix to zero
    Am=A;Bm=B;Cm=C;Dm=D;Em=E;
    % For stability the following is preferred to A = A\E
%     sis = dss(A,B,C,D,E); %create descriptor state space model
%     sys = ss(sis,'explicit'); %convert to continuous state-space model
%     A=sys.A;C=sys.C;
    A = pinv(Em)*Am;
    [Ve,D]=eig(A);              %Compute eigenvalue problem from matrix A
    [w,Id] = sort(abs(diag(D))) ;
    Di = diag(D);
    z=-real(Di(Id))./abs(Di(Id));
    PHI=C(:,Id)*Ve(Id,Id);                    %Modal shapes
    mod=abs(PHI);                             %Modal shape normalisation
    mod=mod./max(mod,[],1);
    ph=angle(PHI);
    ph1=ph - ph(1,:);                           %Phase is in radians

    %Assemble result matrix, convert from rad/s to Hz
    DUMMY=[w';z';mod.*sign(cos(ph1))];
    index = sort(r_iter,'descend');
    id(index(ij)).ident=DUMMY(:,2:2:size(DUMMY,2));
    id(index(ij)).order=k;
    % arrange output of original descriptor state space matrices
    model(index(ij)).A=Am;
    model(index(ij)).B=Bm;
    model(index(ij)).C=Cm;
    model(index(ij)).D=Dm;
    model(index(ij)).E=Em;
    % output of the fitted FRF
    for iij = 1:length(si)
        fit(index(ij)).FRF_fit(:,:,iij) = Cm*( (si(iij).*Em-Am)\Bm);
    end
end

end
