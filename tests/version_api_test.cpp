#include "edge-dit.h"

#include <cstring>
#include <iostream>

int main() {
    if (std::strcmp(ed_version_string(), "0.1.0") != 0) {
        std::cerr << "unexpected version string: " << ed_version_string() << "\n";
        return 1;
    }
    if (ed_version_major() != 0 || ed_version_minor() != 1 || ed_version_patch() != 0) {
        std::cerr << "unexpected version tuple: "
                  << ed_version_major() << "."
                  << ed_version_minor() << "."
                  << ed_version_patch() << "\n";
        return 1;
    }
    return 0;
}
