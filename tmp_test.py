# import interface.sac.config as sac_conf
# import getpass
import os
import sys
import pkgutil

folder = os.getcwd()
print(folder)
print(sys.path)
print("Checking interface")
import interface
package = interface
for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
    print("Found submodule %s (is a package: %s)" % (modname, ispkg))

print("Checking sac")
import interface.sac
package = interface.sac
for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
    print("Found submodule %s (is a package: %s)" % (modname, ispkg))

import pkgutil
for importer, modname, ispkg in pkgutil.walk_packages(path=".", onerror=lambda x: None):
    print(modname)
# import sac
# import sac.config
# import interface.sac
# import interface.sac.config




