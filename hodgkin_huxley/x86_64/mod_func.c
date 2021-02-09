#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _HH_traub_reg(void);
extern void _IL_gutnick_reg(void);
extern void _IM_cortex_reg(void);
extern void _IT_huguenard_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," \"HH_traub.mod\"");
    fprintf(stderr," \"IL_gutnick.mod\"");
    fprintf(stderr," \"IM_cortex.mod\"");
    fprintf(stderr," \"IT_huguenard.mod\"");
    fprintf(stderr, "\n");
  }
  _HH_traub_reg();
  _IL_gutnick_reg();
  _IM_cortex_reg();
  _IT_huguenard_reg();
}
