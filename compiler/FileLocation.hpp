/*
  Copyright 2014 SINTEF ICT, Applied Mathematics.

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef OPM_FILELOCATION_HEADER_INCLUDED
#define OPM_FILELOCATION_HEADER_INCLUDED

class FileLocation
{
public:
    FileLocation()
        : firstline_(-1),
          lastline_(-1)
    {
    }
    explicit FileLocation(const int line)
        : firstline_(line),
          lastline_(line)
    {
    }
    FileLocation(const int firstline, const int lastline)
        : firstline_(firstline),
          lastline_(lastline)
    {
    }
    int firstLine() const
    {
        return firstline_;
    }
    int lastLine() const
    {
        return lastline_;
    }

private:
    int firstline_;
    int lastline_;
};

#endif // OPM_FILELOCATION_HEADER_INCLUDED
