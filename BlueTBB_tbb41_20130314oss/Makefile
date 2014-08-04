# Copyright 2005-2013 Intel Corporation.  All Rights Reserved.
#
# This file is part of Threading Building Blocks.
#
# Threading Building Blocks is free software; you can redistribute it
# and/or modify it under the terms of the GNU General Public License
# version 2 as published by the Free Software Foundation.
#
# Threading Building Blocks is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Threading Building Blocks; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
#
# As a special exception, you may use this file as part of a free software
# library without restriction.  Specifically, if other files instantiate
# templates or use macros or inline functions from this file, or you compile
# this file and link it with other files to produce an executable, this
# file does not by itself cause the resulting executable to be covered by
# the GNU General Public License.  This exception does not however
# invalidate any other reasons why the executable file might be covered by
# the GNU General Public License.

# Note: Unless $(cfg) is specified as either debug or release, both configurations are used.
#       (There is no error detection: any unexpected value is silently handled as if $(cfg) were not specified.)

# Note: Parallel builds, e.g., "make -j 8 cfg=release test", may significantly decrease build time,
#       even though not all potential parallelism has been exposed (there are still artificial "barriers").
#       Conversely, user discipline is required, mostly when multiple targets are requested,
#       to prevent competition between some building and running (normally not an issue, and fairly easily spotted).

tbb_root?=.
include $(tbb_root)/build/common.inc

.PHONY: default build_al build_all all version_string.ver tbb tbbmalloc tbbproxy build_test test rml build_examples examples


# Note that examples currently repeats the compilations of build_examples,
# so build_al (build almost all) is provided as a temporary workaround.
default:   tbb tbbmalloc $(if $(use_proxy),tbbproxy)
build_al:  tbb tbbmalloc $(if $(use_proxy),tbbproxy) build_test
build_all: tbb tbbmalloc $(if $(use_proxy),tbbproxy) build_test build_examples
all:       tbb tbbmalloc $(if $(use_proxy),tbbproxy)       test       examples

# Targets tbb, tbbmalloc, tbbproxy all depend on a common version_string.ver.
version_string.ver: mkdir
ifneq (release,$(cfg))
	$(MAKE) -C "$(work_dir)_debug"    -r -f $(tbb_root)/build/Makefile.tbb       cfg=debug   tbb_root=$(tbb_root) version_string.ver
endif
ifneq (debug,$(cfg))
	$(MAKE) -C "$(work_dir)_release"  -r -f $(tbb_root)/build/Makefile.tbb       cfg=release tbb_root=$(tbb_root) version_string.ver
endif

tbb: version_string.ver
ifneq (release,$(cfg))
	$(MAKE) -C "$(work_dir)_debug"    -r -f $(tbb_root)/build/Makefile.tbb       cfg=debug   tbb_root=$(tbb_root) 
endif
ifneq (debug,$(cfg))
	$(MAKE) -C "$(work_dir)_release"  -r -f $(tbb_root)/build/Makefile.tbb       cfg=release tbb_root=$(tbb_root) 
endif

tbbmalloc: version_string.ver
ifneq (release,$(cfg))
	$(MAKE) -C "$(work_dir)_debug"    -r -f $(tbb_root)/build/Makefile.tbbmalloc cfg=debug   tbb_root=$(tbb_root) malloc
endif
ifneq (debug,$(cfg))
	$(MAKE) -C "$(work_dir)_release"  -r -f $(tbb_root)/build/Makefile.tbbmalloc cfg=release tbb_root=$(tbb_root) malloc
endif

tbbproxy: version_string.ver
ifneq (release,$(cfg))
	$(MAKE) -C "$(work_dir)_debug"    -r -f $(tbb_root)/build/Makefile.tbbproxy  cfg=debug   tbb_root=$(tbb_root) tbbproxy
endif
ifneq (debug,$(cfg))
	$(MAKE) -C "$(work_dir)_release"  -r -f $(tbb_root)/build/Makefile.tbbproxy  cfg=release tbb_root=$(tbb_root) tbbproxy
endif

build_test: tbb tbbmalloc $(if $(use_proxy),tbbproxy)
ifneq (release,$(cfg))
	-$(MAKE) -C "$(work_dir)_debug"   -r -f $(tbb_root)/build/Makefile.tbbmalloc cfg=debug   tbb_root=$(tbb_root) build_malloc_test_no_depends
	-$(MAKE) -C "$(work_dir)_debug"   -r -f $(tbb_root)/build/Makefile.test      cfg=debug   tbb_root=$(tbb_root) build
endif
ifneq (debug,$(cfg))
	-$(MAKE) -C "$(work_dir)_release" -r -f $(tbb_root)/build/Makefile.tbbmalloc cfg=release tbb_root=$(tbb_root) build_malloc_test_no_depends
	-$(MAKE) -C "$(work_dir)_release" -r -f $(tbb_root)/build/Makefile.test      cfg=release tbb_root=$(tbb_root) build
endif

# Note: build_test is not a dependency, to avoid breaking with traditional behavior.
# Note: "-j 1" prevents competition between sets of tests (for CPU use and user attention alike).
test: tbb tbbmalloc $(if $(use_proxy),tbbproxy)
ifneq (release,$(cfg))
	-$(MAKE)      -C "$(work_dir)_debug"   -r -f $(tbb_root)/build/Makefile.tbbmalloc cfg=debug   tbb_root=$(tbb_root) build_malloc_test_no_depends
	-$(MAKE) -j 1 -C "$(work_dir)_debug"   -r -f $(tbb_root)/build/Makefile.tbbmalloc cfg=debug   tbb_root=$(tbb_root) malloc_test
	-$(MAKE)      -C "$(work_dir)_debug"   -r -f $(tbb_root)/build/Makefile.test      cfg=debug   tbb_root=$(tbb_root) build
	-$(MAKE) -j 1 -C "$(work_dir)_debug"   -r -f $(tbb_root)/build/Makefile.test      cfg=debug   tbb_root=$(tbb_root) 
endif
ifneq (debug,$(cfg))
	-$(MAKE)      -C "$(work_dir)_release" -r -f $(tbb_root)/build/Makefile.tbbmalloc cfg=release tbb_root=$(tbb_root) build_malloc_test_no_depends
	-$(MAKE) -j 1 -C "$(work_dir)_release" -r -f $(tbb_root)/build/Makefile.tbbmalloc cfg=release tbb_root=$(tbb_root) malloc_test
	-$(MAKE)      -C "$(work_dir)_release" -r -f $(tbb_root)/build/Makefile.test      cfg=release tbb_root=$(tbb_root) build
	-$(MAKE) -j 1 -C "$(work_dir)_release" -r -f $(tbb_root)/build/Makefile.test      cfg=release tbb_root=$(tbb_root) 
endif

# Needed unless build_test becomes a dependency of test.
.NOTPARALLEL: build_test test

rml: mkdir
ifneq (release,$(cfg))
	$(MAKE) -C "$(work_dir)_debug"    -r -f $(tbb_root)/build/Makefile.rml       cfg=debug   tbb_root=$(tbb_root) 
endif
ifneq (debug,$(cfg))
	$(MAKE) -C "$(work_dir)_release"  -r -f $(tbb_root)/build/Makefile.rml       cfg=release tbb_root=$(tbb_root) 
endif

# TODO: With the new $(cfg) setting, maybe have debug examples again?

# Note: build_examples currently requires a serial build
#       (perhaps because of unexpressed dependencies, or conflicts).
build_examples: tbb tbbmalloc
ifneq (debug,$(cfg))
	$(MAKE) -j 1 -C examples -r -f Makefile tbb_root=.. release
endif

# Note: build_examples is not a dependency, because the executables are currently rebuilt anyway.
# Note: "-j 1" prevents competition between examples (for CPU use and user attention alike).
examples: tbb tbbmalloc
ifneq (debug,$(cfg))
	$(MAKE) -j 1 -C examples -r -f Makefile tbb_root=.. release test
endif

# Needed unless build_examples becomes a dependency of examples.
.NOTPARALLEL: build_examples examples

# Note: This is not a complete substitute for user discipline.
.NOTPARALLEL: build_test examples
.NOTPARALLEL: build_examples test
.NOTPARALLEL: test examples

.PHONY: clean clean_examples mkdir info

# TODO: Or consider release files, debug files, release directory, debug directory,
#       as a (pedantic?) reverse-order teardown.
clean: clean_examples
ifneq (release,$(cfg))
ifeq (macos,$(tbb_os))
	$(shell $(RM) -r $(work_dir)_debug$(SLASH)*.dSYM >$(NUL) 2>$(NUL))
endif
	$(shell $(RM)    $(work_dir)_debug$(SLASH)*.*    >$(NUL) 2>$(NUL))
	$(shell $(RD)    $(work_dir)_debug               >$(NUL) 2>$(NUL))
endif
ifneq (debug,$(cfg))
	$(shell $(RM)    $(work_dir)_release$(SLASH)*.*  >$(NUL) 2>$(NUL))
	$(shell $(RD)    $(work_dir)_release             >$(NUL) 2>$(NUL))
endif
	@echo clean done

clean_examples:
ifneq (debug,$(cfg))
	$(shell $(MAKE) -s -i -r -C examples -f Makefile tbb_root=.. clean >$(NUL) 2>$(NUL))
endif

mkdir:
ifneq (release,$(cfg))
	$(shell $(MD) "$(work_dir)_debug" >$(NUL) 2>$(NUL))
endif
ifneq (debug,$(cfg))
	$(shell $(MD) "$(work_dir)_release" >$(NUL) 2>$(NUL))
endif
	@echo Created $(work_dir)_debug and/or ..._release directories

info:
	@echo OS: $(tbb_os)
	@echo arch=$(arch)
	@echo compiler=$(compiler)
	@echo runtime=$(runtime)
	@echo tbb_build_prefix=$(tbb_build_prefix)
