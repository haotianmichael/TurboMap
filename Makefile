CFLAGS_NDEBUG = -DNDEBUG -O3 
CDEBUG_FLAGS= -g -O0 #-Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -Wno-sign-compare -Wno-unused-function -Wno-c++17-extensions -Wno-\#warnings #-O0 -DNDEBUG
CPPFLAGS=	-DHAVE_KALLOC -D__AMD_SPLIT_KERNELS__ # -Wno-unused-but-set-variable -Wno-unused-variable
CPPFLAGS+= 	$(if $(MAX_MICRO_BATCH),-DMAX_MICRO_BATCH=\($(MAX_MICRO_BATCH)\))
INCLUDES=	-I .
OBJS=		kthread.o kalloc.o misc.o bseq.o sketch.o sdust.o options.o index.o \
			lchain.o align.o hit.o seed.o map.o format.o pe.o esterr.o splitidx.o \
			ksw2_ll_sse.o
# PROG=		minimap2-zerobranch-debug
# PROG=		minimap2-nobalance-debug
PROG=		minimap2$(SUFFIX)
PROG_EXTRA=	sdust minimap2-lite
LIBS=		-lm -lz -lpthread 

ifeq ($(arm_neon),) # if arm_neon is not defined
ifeq ($(sse2only),) # if sse2only is not defined
	OBJS+=ksw2_extz2_sse41.o ksw2_extd2_sse41.o ksw2_exts2_sse41.o ksw2_extz2_sse2.o ksw2_extd2_sse2.o ksw2_exts2_sse2.o ksw2_dispatch.o
else                # if sse2only is defined
	OBJS+=ksw2_extz2_sse.o ksw2_extd2_sse.o ksw2_exts2_sse.o
endif
else				# if arm_neon is defined
	OBJS+=ksw2_extz2_neon.o ksw2_extd2_neon.o ksw2_exts2_neon.o
    INCLUDES+=-Isse2neon
ifeq ($(aarch64),)	#if aarch64 is not defined
	CFLAGS+=-D_FILE_OFFSET_BITS=64 -mfpu=neon -fsigned-char
else				#if aarch64 is defined
	CFLAGS+=-D_FILE_OFFSET_BITS=64 -fsigned-char
endif
endif

ifneq ($(asan),)
	CFLAGS+=-fsanitize=address
	LIBS+=-fsanitize=address
endif

ifneq ($(tsan),)
	CFLAGS+=-fsanitize=thread
	LIBS+=-fsanitize=thread
endif


# turn on debug flags 
ifeq ($(DEBUG),info) 
	CFLAGS = $(CFLAGS_NDEBUG)
	CFLAGS += -DDEBUG_PRINT
else ifeq ($(DEBUG), analyze) 
	CFLAGS = $(CFLAGS_NDEBUG) 
	CFLAGS += -DDEBUG_CHECK -DDEBUG_PRINT
else ifeq ($(DEBUG), verbose)
	CFLAGS = $(CDEBUG_FLAGS) 
	CFLAGS += -DDEBUG_CHECK -DDEBUG_PRINT -DDEBUG_VERBOSE
else 
	CFLAGS = $(CFLAGS_NDEBUG)
endif

.PHONY:all extra clean depend # profile
.SUFFIXES:.c .o

.c.o:
		$(CC) -c $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $< -o $@

all:$(PROG)

extra:all $(PROG_EXTRA)

# build cJSON
CJSON_OBJ= 	cJSON/cJSON.o
INCLUDES += -I cJSON
$(CJSON_OBJ): 
	make -C cJSON

# build kernel objs
include gpu/gpu.mk


# compile with nvcc/hipcc
$(PROG):main.o libminimap2.a
		$(GPU_CC) $(CFLAGS) $(GPU_FLAGS) main.o -o $@ -L. -lminimap2 $(LIBS)

minimap2-lite:example.o libminimap2.a
		$(GPU_CC) $(CFLAGS)  $(GPU_FLAGS) $< -o $@ -L. -lminimap2 $(LIBS)

libminimap2.a:$(OBJS) $(CU_OBJS) $(CJSON_OBJ)
		$(AR) -csru $@ $^

sdust:sdust.c kalloc.o kalloc.h kdq.h kvec.h kseq.h ketopt.h sdust.h
		$(CC) -D_SDUST_MAIN $(CFLAGS) $< kalloc.o -o $@ -lz

# SSE-specific targets on x86/x86_64

ifeq ($(arm_neon),)   # if arm_neon is defined, compile this target with the default setting (i.e. no -msse2)
ksw2_ll_sse.o:ksw2_ll_sse.c ksw2.h kalloc.h
		$(CC) -c $(CFLAGS) -msse2 $(CPPFLAGS) $(INCLUDES) $< -o $@
endif

ksw2_extz2_sse41.o:ksw2_extz2_sse.c ksw2.h kalloc.h
		$(CC) -c $(CFLAGS) -msse4.1 $(CPPFLAGS) -DKSW_CPU_DISPATCH $(INCLUDES) $< -o $@

ksw2_extz2_sse2.o:ksw2_extz2_sse.c ksw2.h kalloc.h
		$(CC) -c $(CFLAGS) -msse2 -mno-sse4.1 $(CPPFLAGS) -DKSW_CPU_DISPATCH -DKSW_SSE2_ONLY $(INCLUDES) $< -o $@

ksw2_extd2_sse41.o:ksw2_extd2_sse.c ksw2.h kalloc.h
		$(CC) -c $(CFLAGS) -msse4.1 $(CPPFLAGS) -DKSW_CPU_DISPATCH $(INCLUDES) $< -o $@

ksw2_extd2_sse2.o:ksw2_extd2_sse.c ksw2.h kalloc.h
		$(CC) -c $(CFLAGS) -msse2 -mno-sse4.1 $(CPPFLAGS) -DKSW_CPU_DISPATCH -DKSW_SSE2_ONLY $(INCLUDES) $< -o $@

ksw2_exts2_sse41.o:ksw2_exts2_sse.c ksw2.h kalloc.h
		$(CC) -c $(CFLAGS) -msse4.1 $(CPPFLAGS) -DKSW_CPU_DISPATCH $(INCLUDES) $< -o $@

ksw2_exts2_sse2.o:ksw2_exts2_sse.c ksw2.h kalloc.h
		$(CC) -c $(CFLAGS) -msse2 -mno-sse4.1 $(CPPFLAGS) -DKSW_CPU_DISPATCH -DKSW_SSE2_ONLY $(INCLUDES) $< -o $@

ksw2_dispatch.o:ksw2_dispatch.c ksw2.h
		$(CC) -c $(CFLAGS) -msse4.1 $(CPPFLAGS) -DKSW_CPU_DISPATCH $(INCLUDES) $< -o $@

# NEON-specific targets on ARM

ksw2_extz2_neon.o:ksw2_extz2_sse.c ksw2.h kalloc.h
		$(CC) -c $(CFLAGS) $(CPPFLAGS) -DKSW_SSE2_ONLY -D__SSE2__ $(INCLUDES) $< -o $@

ksw2_extd2_neon.o:ksw2_extd2_sse.c ksw2.h kalloc.h
		$(CC) -c $(CFLAGS) $(CPPFLAGS) -DKSW_SSE2_ONLY -D__SSE2__ $(INCLUDES) $< -o $@

ksw2_exts2_neon.o:ksw2_exts2_sse.c ksw2.h kalloc.h
		$(CC) -c $(CFLAGS) $(CPPFLAGS) -DKSW_SSE2_ONLY -D__SSE2__ $(INCLUDES) $< -o $@

# other non-file targets

clean: cleangpu
		rm -fr gmon.out *.o a.out $(PROG) $(PROG_EXTRA) *~ *.a *.dSYM build dist mappy*.so mappy.c python/mappy.c mappy.egg*

depend:
		(LC_ALL=C; export LC_ALL; makedepend -Y -- $(CFLAGS) $(CPPFLAGS) -- *.c)

# DO NOT DELETE

align.o: minimap.h mmpriv.h bseq.h kseq.h ksw2.h kalloc.h
bseq.o: bseq.h kvec.h kalloc.h kseq.h
esterr.o: mmpriv.h minimap.h bseq.h kseq.h
example.o: minimap.h kseq.h
format.o: kalloc.h mmpriv.h minimap.h bseq.h kseq.h
hit.o: mmpriv.h minimap.h bseq.h kseq.h kalloc.h khash.h
index.o: kthread.h bseq.h minimap.h mmpriv.h kseq.h kvec.h kalloc.h khash.h
index.o: ksort.h
kalloc.o: kalloc.h
ksw2_extd2_sse.o: ksw2.h kalloc.h
ksw2_exts2_sse.o: ksw2.h kalloc.h
ksw2_extz2_sse.o: ksw2.h kalloc.h
ksw2_ll_sse.o: ksw2.h kalloc.h
kthread.o: kthread.h
lchain.o: mmpriv.h minimap.h bseq.h kseq.h kalloc.h krmq.h
main.o: bseq.h minimap.h mmpriv.h kseq.h ketopt.h
map.o: kthread.h kvec.h kalloc.h sdust.h mmpriv.h minimap.h bseq.h kseq.h
map.o: khash.h ksort.h
misc.o: mmpriv.h minimap.h bseq.h kseq.h ksort.h
options.o: mmpriv.h minimap.h bseq.h kseq.h
pe.o: mmpriv.h minimap.h bseq.h kseq.h kvec.h kalloc.h ksort.h
sdust.o: kalloc.h kdq.h kvec.h sdust.h
seed.o: mmpriv.h minimap.h bseq.h kseq.h kalloc.h ksort.h
sketch.o: kvec.h kalloc.h mmpriv.h minimap.h bseq.h kseq.h
splitidx.o: mmpriv.h minimap.h bseq.h kseq.h