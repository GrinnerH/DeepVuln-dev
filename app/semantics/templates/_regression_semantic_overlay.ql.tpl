/**
 * @kind diagnostic
 * @id deepvuln/semantic-overlay
 */
import ProjectSemantics
import semmle.code.cpp.security.BufferAccess

predicate isOverlayWrapper(BufferAccess ba) {
  // __OVERLAY_ACCESS_PRED__
}

from BufferAccess ba
where isOverlayWrapper(ba)
select ba, "MODELED_WRAPPER_CALL"
