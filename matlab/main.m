function varargout = main(varargin)

originalPath = path;
pathCleanup = onCleanup(@() path(originalPath));
addpath(fullfile(fileparts(mfilename('fullpath')), 'model'));
[varargout{1:nargout}] = run_model(varargin{:});

end
