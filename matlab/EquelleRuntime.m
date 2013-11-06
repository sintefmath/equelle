classdef EquelleRuntime < handle
    properties
        grid;
        input;
    end
    methods
        function obj = EquelleRuntime(g)
            obj.grid = g;
        end
        
        function setInput(obj, input)
            obj.input = input;
        end

        function ret = AllCells(obj)
            ret = [1:obj.grid.cells.num]';
        end
        
        function ret = AllFaces(obj)
            ret = [1:obj.grid.faces.num]';
        end
        
        function ret = Normal(obj, faces)
            n = obj.grid.faces.normals(faces,:);
            % Must normalize mrst normals
            len = sqrt(sum(n.*n, 2));
            ret = n./repmat(len, [1, size(n, 2)]);
        end
        
        function ret = Dot(~, v1, v2)
            ret = sum(v1.*v2, 2);
        end
        
        function ret = UserSpecifiedScalarWithDefault(obj, tag, default)
            if (isfield(obj.input, tag))
                ret = getfield(obj.input, tag); %#ok<GFLD>
            else
                ret = default;
            end
        end
        
        function ret = UserSpecifiedCollectionOfScalar(obj, tag, domain)
            if (~isfield(obj.input, tag))
                disp(sprintf('Missing field in input: %s', tag));
                error('You need to specify input containing the missing field.')
            end
            data = getfield(obj.input, tag); %#ok<GFLD>
            assert(all(size(data) == size(domain)))
            ret = data;
        end
        
        function Output(~, string, data)
            disp(string)
            disp(data)
        end
    end
end