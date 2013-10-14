;;; equelle-mode-el -- Major mode for editing Equelle files

;; Author: Scott Andrew Borton <scott@pp.htv.fi>
;; Author: Atgeirr F. Rasmussen <atgeirr@sintef.no>
;; Created: 25 Sep 2000
;; Modified: 23 Oct 2013
;; Keywords: Equelle major-mode

;; Copyright (C) 2000, 2003 Scott Andrew Borton <scott@pp.htv.fi>
;; Copyright (C) 2013 SINTEF ICT, Applied Mathematics

;; This program is free software; you can redistribute it and/or
;; modify it under the terms of the GNU General Public License as
;; published by the Free Software Foundation; either version 2 of
;; the License, or (at your option) any later version.

;; This program is distributed in the hope that it will be
;; useful, but WITHOUT ANY WARRANTY; without even the implied
;; warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
;; PURPOSE.  See the GNU General Public License for more details.

;; You should have received a copy of the GNU General Public
;; License along with this program; if not, write to the Free
;; Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
;; MA 02111-1307 USA

;;; Commentary:
;; 
;; This mode started from an example used in a tutorial about Emacs
;; mode creation. The tutorial can be found here:
;; http://two-wugs.net/emacs/mode-tutorial.html

;;; Code:
(defvar equelle-mode-hook nil)
(defvar equelle-mode-map
  (let ((equelle-mode-map (make-keymap)))
    (define-key equelle-mode-map "\C-j" 'newline-and-indent)
    equelle-mode-map)
  "Keymap for Equelle major mode")

(add-to-list 'auto-mode-alist '("\\.equelle\\'" . equelle-mode))

(defconst equelle-font-lock-keywords-1
  (list
   '("\\<\\(Collection\\|Of\\|On\\|Subset\\|Scalar\\|Vector\\|Bool\\|Cell\\|Face\\|Edge\\|Vertex\\|Function\\|And\\|Or\\|Not\\|Xor\\)\\>" . font-lock-keyword-face)
   '("\\('\\w*'\\)" . font-lock-variable-name-face))
  "Minimal highlighting expressions for Equelle mode.")

(defconst equelle-font-lock-keywords-2
  (append equelle-font-lock-keywords-1
		  (list
		   '("\\<\\(InteriorCells\\|BoundaryCells\\|AllCells\\|InteriorFaces\\|BoundaryFaces\\|AllFaces\\|InteriorEdges\\|BoundaryEdges\\|AllEdges\\|InteriorVertices\\|BoundaryVertices\\|AllVertices\\|FirstCell\\|SeconCell\\|IsEmpty\\|Centroid\\|Normal\\|UserSpecifiedScalarWithDefault\\|UserSpecifiedCollectionOfScalar\\|UserSpecifiedCollectionOfFaceSubsetOf\\|Gradient\\|Divergence\\|NewtonSolve\\|Output\\)\\>" . font-lock-builtin-face)
		   '("\\<\\(True\\|False\\)\\>" . font-lock-constant-face)))
  "Additional Keywords to highlight in Equelle mode.")

(defconst equelle-font-lock-keywords-3
  (append equelle-font-lock-keywords-2
		  (list
		 ; These are some possible built-in values for Equelle attributes
			 ; "ROLE" "ORGANISATIONAL_UNIT" "STRING" "REFERENCE" "AND"
			 ; "XOR" "WORKFLOW" "SYNCHR" "NO" "APPLICATIONS" "BOOLEAN"
							 ; "INTEGER" "HUMAN" "UNDER_REVISION" "OR"
		   '("\\<\\(A\\(ND\\|PPLICATIONS\\)\\|BOOLEAN\\|HUMAN\\|INTEGER\\|NO\\|OR\\(GANISATIONAL_UNIT\\)?\\|R\\(EFERENCE\\|OLE\\)\\|S\\(TRING\\|YNCHR\\)\\|UNDER_REVISION\\|WORKFLOW\\|XOR\\)\\>" . font-lock-constant-face)))
  "Balls-out highlighting in Equelle mode.")

(defvar equelle-font-lock-keywords equelle-font-lock-keywords-3
  "Default highlighting expressions for Equelle mode.")

(defun equelle-indent-line ()
  "Indent current line as Equelle code."
  (interactive)
  (beginning-of-line)
  (if (bobp)
	  (indent-line-to 0)		   ; First line is always non-indented
	(let ((not-indented t) cur-indent)
	  (if (looking-at "^[ \t]*END_") ; If the line we are looking at is the end of a block, then decrease the indentation
		  (progn
			(save-excursion
			  (forward-line -1)
			  (setq cur-indent (- (current-indentation) default-tab-width)))
			(if (< cur-indent 0) ; We can't indent past the left margin
				(setq cur-indent 0)))
		(save-excursion
		  (while not-indented ; Iterate backwards until we find an indentation hint
			(forward-line -1)
			(if (looking-at "^[ \t]*END_") ; This hint indicates that we need to indent at the level of the END_ token
				(progn
				  (setq cur-indent (current-indentation))
				  (setq not-indented nil))
			  (if (looking-at "^[ \t]*\\(PARTICIPANT\\|MODEL\\|APPLICATION\\|WORKFLOW\\|ACTIVITY\\|DATA\\|TOOL_LIST\\|TRANSITION\\)") ; This hint indicates that we need to indent an extra level
				  (progn
					(setq cur-indent (+ (current-indentation) default-tab-width)) ; Do the actual indenting
					(setq not-indented nil))
				(if (bobp)
					(setq not-indented nil)))))))
	  (if cur-indent
		  (indent-line-to cur-indent)
		(indent-line-to 0))))) ; If we didn't see an indentation hint, then allow no indentation

(defvar equelle-mode-syntax-table
  (let ((equelle-mode-syntax-table (make-syntax-table)))
	
    ; This is added so entity names with underscores can be more easily parsed
	(modify-syntax-entry ?_ "w" equelle-mode-syntax-table)
	
	; Comments start with # and go till e.o.l.
        (modify-syntax-entry ?# "< b" equelle-mode-syntax-table)
        (modify-syntax-entry ?\n "> b" equelle-mode-syntax-table)
	equelle-mode-syntax-table)
  "Syntax table for equelle-mode")
  
(defun equelle-mode ()
  (interactive)
  (kill-all-local-variables)
  (use-local-map equelle-mode-map)
  (set-syntax-table equelle-mode-syntax-table)
  ;; Set up font-lock
  (set (make-local-variable 'font-lock-defaults) '(equelle-font-lock-keywords))
  ;; Register our indentation function
  (set (make-local-variable 'indent-line-function) 'equelle-indent-line)  
  (setq major-mode 'equelle-mode)
  (setq mode-name "Equelle")
  (run-hooks 'equelle-mode-hook))

(provide 'equelle-mode)

;;; equelle-mode.el ends here



