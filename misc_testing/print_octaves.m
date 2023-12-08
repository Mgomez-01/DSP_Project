function print_octaves(Octaves,n,fs)
A4 = 440;
C4 = A4.*2.^(-9./12);
B4 = A4.*2.^(2./12);
Cs = C4.*Octaves;
Bs = B4.*Octaves;

Centers = (Cs + Bs)./2;
cell(size(Centers));
octave_array = arrayfun(@(x) sprintf('Octave %d', x), n, 'UniformOutput', false);

w_Cs = Cs.*2.*pi./fs;
w_Bs = Bs.*2.*pi./fs;
w_Centers = Centers.*2.*pi./fs;
rows = {'Lower (Hz)','Lower (Rad)','Upper (Hz)','Upper (Rad)','Center (Hz)','Center (Rad)'};

% Summarize data in a table
T = array2table([Cs; w_Cs; Bs; w_Bs; Centers; w_Centers],'VariableNames',octave_array,'RowName',rows);
disp(T)
disp('Hz are not normalized.')
disp('Radians are normalized by sampling frequency.')

end