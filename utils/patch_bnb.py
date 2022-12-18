"""
Copyright [2022] Victor C Hall

Licensed under the GNU Affero General Public License;
You may not use this code except in compliance with the License.
You may obtain a copy of the License at

    https://www.gnu.org/licenses/agpl-3.0.en.html

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# see: https://github.com/TimDettmers/bitsandbytes/issues/30 for explanation
import sys
import os
from subprocess import check_output
import shutil

_CEXT_PATCH = "            self.lib = ct.cdll.LoadLibrary(str(binary_path))"
_MAIN_PATCH = "    return 'libbitsandbytes_cuda116.dll'"

def patch_main():
    bnbpath_main = "venv/Lib/site-packages/bitsandbytes/cuda_setup/main.py"
    try: 
        with open(bnbpath_main, "r") as f:
            contents = f.read()
            contents = contents.split('\n')
    except Exception as ex:
        print(f"cannot find bitsandbytes install, aborting, error: {ex}")
        return False

    main_patched = False

    for i, line in enumerate(contents):
        if i == 112:
            if line != _MAIN_PATCH:
                contents[i] = _MAIN_PATCH
                main_patched = True
            else:
                print(" *** Already patched!")
                main_patched = True

    assert main_patched, "unable to patch bitsandbytes, may be mismatched version, requires 0.35.0"

    with open(bnbpath_main, "w") as f:
        for line in contents:
            f.write(line + "\n")
        #print(contents)
    
    return main_patched

def patch_cext():    
    bnbpath_cextension = "venv/Lib/site-packages/bitsandbytes/cextension.py"
    try: 
        with open(bnbpath_cextension, "r") as f:
            contents = f.read()
            contents = contents.split('\n')
    except Exception as ex:
        print(f"cannot find bitsandbytes install, aborting, error: {ex}")
        return False

    cext_patched = False

    for i, line in enumerate(contents):
        # update both lines 28 and 31 to be sure correct dll is returned
        if (i == 30 or i == 27):
            if line != _CEXT_PATCH:
                contents[i] = _CEXT_PATCH
                cext_patched = True
            else:
                cext_patched = True

    assert cext_patched, "unable to patch bitsandbytes, died midprocess, something broke and may need to reinstall bitsandbytes==0.35.0"

    with open(bnbpath_cextension, "w") as f:
        for line in contents:
            f.write(line + "\n")
        #print(contents)

    return cext_patched

def iswindows():
    return sys.platform.startswith('win')

def error():
    print("Somethnig went wrong trying to patch bitsandbytes, aborting")
    print("make sure your venv is activated and try again")
    print("or if activated try: ")
    print("    pip install bitsandbytes==0.35.0")
    raise RuntimeError("** FATAL ERROR: unable to patch bitsandbytes for Windows env")

def check_dlls():
    dll_exists = os.path.exists("venv/Lib/site-packages/bitsandbytes/libbitsandbytes_cuda116.dll")
    if not dll_exists:
        if not os.path.exists("tmp/bnb_cache"):
            check_output("git clone https://github.com/DeXtmL/bitsandbytes-win-prebuilt tmp/bnb_cache", shell=True)
        shutil.copy("tmp/bnb_cache/libbitsandbytes_cuda116.dll", "venv/Lib/site-packages/bitsandbytes/libbitsandbytes_cuda116.dll")
    dll_exists = os.path.exists("venv/Lib/site-packages/bitsandbytes/libbitsandbytes_cuda116.dll")
    return dll_exists

def main():
    """
    applies a patch for windows compatibility for bitsandbytes 0.35.0 for using their AdamW8bit optimizer
    """
    if iswindows():
        print()
        print(" *** Applying bitsandbytes patch for windows ***")
        if not check_dlls():
            print("unable to find bitsandbytes dll or clone them from git, aborting")
            raise RuntimeError("** FATAL ERROR: unable to patch bitsandbytes for Windows env")

        main_patched = patch_main()
        cext_patched = patch_cext()
        if main_patched and cext_patched:
            try:
                print(" *************************************************************")
                print(" *** bitsandbytes windows patch applied, attempting import *** ")
                import bitsandbytes
                print(f" *** bitsandbytes patch succeeded, everything looks good!  ***")
            except:
                error()
        else: 
            error()
    else:
        print(" *** not using windows environment, skipping bitsandbytes patch ***")
        return

if __name__ == "__main__":
    main()
